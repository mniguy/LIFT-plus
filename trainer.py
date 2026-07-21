import os
import sys
import time
import datetime
import math

from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from clip import clip
from timm.models.vision_transformer import (
    vit_base_patch16_224, vit_base_patch16_384,
    vit_large_patch16_224, vit_large_patch16_384,
)
from timm.models.resnet import resnet50, resnet101, resnet152

from models import PEFT_Model
import datasets

from utils.evaluator import Evaluator
from utils.losses import *
from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.transforms import *


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.build_model()
        self.build_data()
        self.build_criterion()
        self.build_tuner()
        if not (cfg.zero_shot or cfg.test_only):
            self.build_optimizer()
        
        """
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        """
    
    def build_model(self):
        cfg = self.cfg
        print("Building model")
        
        # load model
        if cfg.backbone.startswith("CLIP-"):
            backbone = cfg.backbone[5:]
            print(f"Loading CLIP (backbone: {backbone})")
            model = clip.load(backbone, device=self.device)[0]
        
        elif cfg.backbone.startswith("IN21K-"):
            backbone = cfg.backbone[6:]
            print(f"Loading IN21K pre-trained model (backbone: {backbone})")
            if backbone == "ViT-B/16":
                model = vit_base_patch16_224(pretrained=True).eval()
            elif backbone == "ViT-B/16@384px":
                model = vit_base_patch16_384(pretrained=True).eval()
            elif backbone == "ViT-L/16":
                model = vit_large_patch16_224(pretrained=True).eval()
            elif backbone == "ViT-L/16@384px":
                model = vit_large_patch16_384(pretrained=True).eval()
            else:
                raise ValueError
        
        elif cfg.backbone.startswith("IN1K-"):
            backbone = cfg.backbone[5:]
            print(f"Loading IN1K pre-trained model (backbone: {backbone})")
            if backbone == "RN50":
                model = resnet50(pretrained=True).eval()
            elif backbone == "RN101":
                model = resnet101(pretrained=True).eval()
            elif backbone == "RN152":
                model = resnet152(pretrained=True).eval()
            else:
                raise ValueError

        else:
            raise ValueError

        self.model = PEFT_Model(model).to(self.device)

        prec = cfg.prec_train
        if prec == "fp16":
            self.model.half()
        elif prec in ["fp32", "amp"]:
            self.model.float()
        else:
            raise ValueError

        model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model params: {model_params}")

    def build_data(self):
        cfg = self.cfg
        resolution = cfg.resolution
        mean = cfg.mean
        std = cfg.std

        print("Building data")

        if cfg.mda:
            transform_train = transforms.Compose([
                MinimalistRandomResizedCrop(resolution, cfg.num_epochs, sched_func=cfg.mda_func, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(resolution, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            expand = cfg.expand if cfg.expand is not None else 24
            transform_test = transforms.Compose([
                transforms.Resize(resolution + expand, interpolation=InterpolationMode.BICUBIC),
                transforms.FiveCrop(resolution),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Normalize(mean, std),
            ])
        else:
            expand = cfg.expand if cfg.expand is not None else resolution // 7
            transform_test = transforms.Compose([
                transforms.Resize(resolution + expand, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(cfg.root, split="train", transform=transform_train)
        init_dataset = getattr(datasets, cfg.dataset)(cfg.root, split="train", transform=transform_plain)
        test_dataset = getattr(datasets, cfg.dataset)(cfg.root, split="test", transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        self.many_classes = (torch.tensor(self.cls_num_list) > 100).nonzero().squeeze()
        self.med_classes = ((torch.tensor(self.cls_num_list) >= 20) & (torch.tensor(self.cls_num_list) <= 100)).nonzero().squeeze()
        self.few_classes = (torch.tensor(self.cls_num_list) < 20).nonzero().squeeze()

        assert cfg.batch_size % cfg.accum_step == 0, "batch_size must be divisible by accum_step"
        micro_batch_size = cfg.batch_size // cfg.accum_step

        self.train_loader = DataLoader(train_dataset,
            batch_size=micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.init_loader = DataLoader(init_dataset,
            batch_size=64, shuffle=False,
            sampler=DownSampler(init_dataset.labels, n_max=100),
            num_workers=cfg.num_workers, pin_memory=True)
        
        self.test_loader = DataLoader(test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
    
        print("Total training points:", sum(self.cls_num_list))
        print("Class frequencies:", self.cls_num_list)

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal":
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM":
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.classifier_scale)
        elif cfg.loss_type == "CB":
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW":
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS":
            self.criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA":
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE":
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "VS":
            self.criterion = VSLoss(cls_num_list=cls_num_list,
                                    gamma=getattr(cfg, "VS_GAMMA", 0.3), tau=getattr(cfg, "VS_TAU", 1.0))
        else:
            raise ValueError
        
        # ---- Text regularizer (Logit KD) ----
        self.text_reg_lambda = float(getattr(cfg, "TEXT_REG_LAMBDA", 0.0))
        
        if self.text_reg_lambda > 0:
            self.kd_criterion = LogitKDLoss(T=float(getattr(cfg, "TEXT_REG_T", 1.0)))
        else:
            self.kd_criterion = None

        # ---- InfoNCE ---
        self.infonce_lambda = float(getattr(cfg, "INFONCE_LAMBDA", 0.0))
        self.infonce_T      = float(getattr(cfg, "INFONCE_T", 0.1))

        if self.infonce_lambda > 0:
            self.infonce_criterion = InfoNCELoss(T=self.infonce_T, reduction="mean")
        else: 
            self.infonce_criterion = None
    
    def _clean_wiki_text(self, txt: str) -> str:
        txt = txt.replace("\ufeff", "")
        txt = re.sub(r"==.*?==", " ", txt)
        txt = re.sub(r"\[[0-9]+\]", "", txt)
        txt = re.sub(r"\s+", " ", txt)

        return txt.strip()

    def _split_sentences(self, txt: str) -> List[str]:
        sents = re.split(r"(?<=[.!?])\s+", txt)

        return [s.strip() for s in sents if len(s.strip()) > 0] 

    def _get_prompt_templates(self) -> List[str]:
        prompt_mode = getattr(self.cfg, "PROMPT_MODE", "default")

        if prompt_mode == "default":
            return ["a photo of a {}."]
        if prompt_mode == "places_scene":
            return ["a photo of a {} scene."]
        if prompt_mode == "places_place":
            return ["a photo of a place called {}."]
        if prompt_mode == "places_ensemble":
            return [
                "a photo of a {}.",
                "a photo of a {} scene.",
                "a photo of a place called {}.",
                "a photo of the inside or outside of a {}.",
            ]

        raise ValueError(f"Unknown PROMPT_MODE: {prompt_mode}")
    
    def build_wiki_corpus(
        self,
        caption_dir: str,
        classnames: List[str],
        max_sentences: int = 0,
        max_chars: int = 0,
        index_map: Dict[int, int] = None,
    ) -> Dict[int, List[str]]:
        corpus = {}
        for i, _ in enumerate(classnames):
            file_id = index_map.get(i, i) if index_map is not None else i
            caption_path = os.path.join(caption_dir, f"desc_{file_id}.txt")
            sents = []
            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8", errors='replace') as f:
                    raw = f.read()
                if max_chars > 0:
                    raw = raw[:max_chars]
                txt = self._clean_wiki_text(raw)
                sents = self._split_sentences(txt)
                if max_sentences > 0 and len(sents) > max_sentences:
                    sents = sents[:max_sentences]
            corpus[i] = sents
        return corpus

    def _wiki_index_map(self):
        """Map class label -> desc_{id}.txt file id.

        iNat's desc files are numbered by categories.json RAW order, but classnames
        are sorted(set(names)); reading desc_{label} loads the wrong species. If a
        categories.json exists (iNat only), remap label -> raw id so the correct
        article is read. Returns None (identity) for ImageNet/Places (no categories.json,
        already aligned). Keeps the raw desc files untouched -- no data swap needed.
        """
        cats_path = os.path.join("datasets", self.cfg.dataset, "categories.json")
        if not os.path.exists(cats_path):
            return None
        import json
        try:
            cats = json.load(open(cats_path))
            names = [c["name"] for c in cats]
        except (ValueError, TypeError, KeyError):
            return None
        name2raw = {}
        for raw_id, nm in enumerate(names):
            name2raw.setdefault(nm, raw_id)
        idx_map = {label: name2raw[cn] for label, cn in enumerate(self.classnames) if cn in name2raw}
        if not idx_map or all(k == v for k, v in idx_map.items()):
            return None  # already identity-aligned
        n_remap = sum(1 for k, v in idx_map.items() if k != v)
        print(f"[Wiki] Remapping desc index via {cats_path}: "
              f"{len(idx_map)}/{len(self.classnames)} classes matched, {n_remap} remapped "
              f"(categories raw-order -> sorted label).")
        return idx_map

    def build_tuner(self):
    
        cfg = self.cfg

        print("Building tuner")

        if cfg.coop:
            if cfg.coop_init is None:
                ctx_len = cfg.coop_ctx_len
                cls_pos = cfg.coop_cls_pos

                if cls_pos == "front":
                    self.template = "{}" + " X" * ctx_len + "."
                    ctx_loc = list(range(-ctx_len - 1, -1))

                elif cls_pos == "middle":
                    ctx_len_left, ctx_len_right = (ctx_len // 2), (ctx_len + 1) // 2
                    self.template = "X " * ctx_len_left + "{}" + " X" * ctx_len_right + "."
                    ctx_loc = list(range(ctx_len_left)) + list(range(-ctx_len_right - 1, -1))

                elif cls_pos == "end":
                    self.template = "X " * ctx_len + "{}."
                    ctx_loc = list(range(ctx_len))
                
                else:
                    raise ValueError
                
                print("Add learnable context with template '{}'.".format(self.template))
                self.model.text_encoder.add_learnable_context(ctx_loc=ctx_loc)
            else:
                if cfg.coop_init == "photo":
                    self.template = "a photo of a {}."
                    ctx_loc = [0, 1, 2, 3, -1]
                else:
                    raise ValueError

                print("Note: Template '{}' is used to initialize the context.".format(self.template),
                      "The context length is {}, and `ctx_len` will be deprecated.".format(len(ctx_loc)))
                self.model.text_encoder.add_learnable_context(ctx_loc=ctx_loc, init_text=self.template)
        else:
            templates = self._get_prompt_templates()
            self.template = templates[0]
            prompt_mode = getattr(cfg, "PROMPT_MODE", "default")
            if len(templates) == 1:
                print("Use template '{}' for prompt generation.".format(self.template))
            else:
                print("Use prompt mode '{}' with templates: {}".format(prompt_mode, templates))

        for _name, _cfg in (("image_encoder", cfg.v), ("text_encoder", cfg.l)):
            if not hasattr(self.model, _name):
                continue
            
            _encoder = getattr(self.model, _name)
            
            def parse_layers(layers):
                n_layers = len(_encoder.blocks)
                if layers is None:
                    return list(range(n_layers))
                elif isinstance(layers, int):
                    return list(range(n_layers - layers, n_layers))
                else:
                    return eval(layers)
            
            if _cfg.fft:
                layers = parse_layers(_cfg.fft_layers)
                print("Fine-tune all parameters in layers {}.".format(layers))
                _encoder.unfreeze_params(layers=layers)

            if _cfg.bitfit:
                layers = parse_layers(_cfg.bitfit_layers)
                print("Fine-tune bias parameters in layers {}.".format(layers))
                _encoder.unfreeze_bias(layers=layers)
            
            if _cfg.pt:
                layers = parse_layers(_cfg.pt_layers)
                prompt_len = _cfg.pt_len or 2 ** max(0, int(math.log2(self.num_classes / (len(layers)))))
                print("Add learnable prompt with length {} to layers {}.".format(prompt_len, layers))
                _encoder.add_learnable_prompt(layers=layers, prompt_len=prompt_len)
            
            if _cfg.lora:
                layers = parse_layers(_cfg.lora_layers)
                bottle_dim = _cfg.lora_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(layers) * 4))))
                print("Add LoRA with bottle dimension {} to layers {}.".format(bottle_dim, layers))
                _encoder.add_lora(layers=layers, bottle_dim=bottle_dim)

            if _cfg.adapter:
                layers = parse_layers(_cfg.adapter_layers)
                bottle_dim = _cfg.adapter_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(layers) * 2))))
                print("Add Adapter with bottle dimension {} to layers {}.".format(bottle_dim, layers))
                _encoder.add_adapter(layers=layers, bottle_dim=bottle_dim)

            if _cfg.adaptformer:
                layers = parse_layers(_cfg.adaptformer_layers)
                bottle_dim = _cfg.adaptformer_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(layers) * 2))))
                print("Add AdaptFormer with bottle dimension {} to layers {}.".format(bottle_dim, layers))
                _encoder.add_adaptformer(layers=layers, bottle_dim=bottle_dim)
            
            if _cfg.ssf:
                layers = parse_layers(_cfg.ssf_layers)
                print("Add SSF to layers {}.".format(layers))
                _encoder.add_ssf(layers=layers)

            if _cfg.aft:
                layers = parse_layers(_cfg.aft_layers)
                loc = _cfg.aft_loc
                seed = _cfg.aft_seed
                if _cfg.aft_ratio is not None:
                    ratio = float(_cfg.aft_ratio)
                elif loc == "attn":
                    ratio = self.num_classes / (len(layers) * (_encoder.embed_dim * 4 + 4))
                elif loc == "mlp":
                    ratio = self.num_classes / (len(layers) * (_encoder.embed_dim * 8 + 5))
                elif loc == "all":
                    ratio = self.num_classes / (len(layers) * (_encoder.embed_dim * 12 + 9))
                else:
                    raise ValueError
                print("Fine-tune a random part of parameters in {} layers {}".format(loc, layers))
                _encoder.add_aft(layers=layers, ratio=ratio, loc=loc, seed=seed)
                
        if cfg.proj_tuning:
            print("Fine-tune the projections of both branches.")
            self.model.unfreeze_image_proj()
            self.model.unfreeze_text_proj()
        
        if cfg.clip_adapter:
            bottle_dim = cfg.clip_adapter_dim
            print("Add CLIP-Adapter with bottle dimension {}.".format(bottle_dim))
            self.model.add_clip_adapter(bottle_dim=bottle_dim)

        if cfg.classifier:
            print("Add classifier on top of the vision model.")
            self.model.add_classifier(cfg.classifier, self.num_classes, scale=cfg.classifier_scale)

            if cfg.classifier == "CosineClassifierGroupScale":
                # #1: fixed per-class scale from frequency (rarest -> GROUP_SCALE_TAIL, most-frequent -> GROUP_SCALE_HEAD)
                cn = torch.as_tensor(self.cls_num_list, dtype=torch.float32)
                rank = torch.argsort(torch.argsort(cn)).float()      # 0 = rarest ... N-1 = most frequent
                rarity = 1.0 - rank / max(len(cn) - 1, 1)            # rarest -> 1, most frequent -> 0
                s_head = float(getattr(cfg, "GROUP_SCALE_HEAD", cfg.classifier_scale))
                s_tail = float(getattr(cfg, "GROUP_SCALE_TAIL", cfg.classifier_scale))
                scale_vec = s_head + (s_tail - s_head) * rarity
                self.model.tuner["classifier"].set_scale_vector(scale_vec)
                print(f"[GroupScale] s_head={s_head} s_tail={s_tail} -> per-class scale "
                      f"[{scale_vec.min().item():.1f}, {scale_vec.max().item():.1f}]")

            if not (cfg.zero_shot or cfg.test_only) and cfg.classifier_init is not None:
                classifier_init = cfg.classifier_init
                
                if classifier_init == "semantic":
                    print("Using semantic-aware initialization.")
                    with torch.no_grad():
                        class_features = self.compute_prompt_class_features()
                        if getattr(cfg, "PROMPT_CENTER", False):  # control + #3: caption-free centering
                            class_features = self._center_prototypes(class_features)
                            print(f"[PROMPT_CENTER] mode={getattr(cfg, 'PROMPT_CENTER_MODE', 'global')} "
                                  f"applied to prototypes.")
                    # store fixed textual prior for optional training-time regularization
                    self.text_prior_weight = F.normalize(class_features, dim=-1).detach()
                    self.model.init_classifier_weight(class_features, feature_modality="text")
                
                elif classifier_init == "hybrid":
                    print("Using real-time hybrid initialization.")
                    with torch.no_grad():
                        class_features = self._compute_caption_features()
                    # store fixed textual prior for optional training-time regularization
                    self.text_prior_weight = class_features.detach()
                    self.model.init_classifier_weight(class_features, feature_modality="text")
                
                elif classifier_init == "class_mean":
                    print("Using class mean feature for initialization.")
                    with torch.no_grad():
                        train_features, train_labels = self.compute_train_features()
                    # compute class means
                    sorted_index = train_labels.argsort()
                    train_features = train_features[sorted_index]
                    train_labels = train_labels[sorted_index]
                    _, label_counts = torch.unique(train_labels, return_counts=True)
                    class_means = torch.stack([x.mean(dim=0) for x in torch.split(train_features, label_counts.tolist())])
                    # initialize classifier
                    self.model.init_classifier_weight(class_means, feature_modality="image")

                elif classifier_init == "linear_probing":
                    print("Using linear probing for initialization.")
                    with torch.no_grad():
                        train_features, train_labels = self.compute_train_features()
                    # compute classifier weights
                    clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced")
                    clf.fit(train_features.cpu(), train_labels.cpu())
                    class_weights = torch.from_numpy(clf.coef_).to(train_features.dtype).to(self.device)
                    class_weights = F.normalize(class_weights, dim=-1)
                    # initialize classifier
                    self.model.init_classifier_weight(class_weights, feature_modality="image")

                else:
                    raise ValueError
                
                torch.cuda.empty_cache()

        if hasattr(self, "text_prior_weight") and getattr(cfg, "PRIOR_REG_MODE", "fixed") == "class_gate":
            self.build_prior_gate()

        self.tuner = self.model.tuner

        # print parameters
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        print(f"Tuned params: {tuned_params}")
        for name, param in self.tuner.named_parameters():
            print(f"├─{name}: {param.numel()}")

    @torch.no_grad()
    def _compute_caption_features(self):
        cfg = self.cfg
        top_k = cfg.HYBRID_TOPK
        device = self.device

        word_chunk_size = cfg.CHUNK_SIZE
        sim_threshold = cfg.SIM_THRESHOLD

        caption_dir = os.path.join("datasets", self.cfg.dataset, 'wiki')
        
        print(f"[Wiki] Building corpus from {caption_dir} ...")
        assert os.path.exists(caption_dir), f"Wiki caption directory not found at: {caption_dir}"

        corpus = self.build_wiki_corpus(
            caption_dir=caption_dir,
            classnames=self.classnames,
            max_sentences=getattr(cfg, "WIKI_MAX_SENTENCES", 0),
            max_chars=getattr(cfg, "WIKI_MAX_CHARS", 0),
            index_map=self._wiki_index_map(),
        )

        print(f"[Wiki] Computing features (top-{top_k}, thresh>{sim_threshold}, dynamic_alpha, chunked) for dataset={cfg.dataset} ...")

        w_prompts_raw = self.compute_prompt_class_features()
        w_prompts_raw = F.normalize(w_prompts_raw, dim=-1)

        # Optional: dump which caption sentences get selected per class (inspection).
        _dump_path = os.environ.get("DUMP_CAPTIONS")
        _dump = [] if _dump_path else None

        # --- experimental caption knobs (#3 geometry, #4 placement); defaults = current behavior ---
        caption_center = bool(getattr(cfg, "CAPTION_CENTER", False))
        caption_blend = getattr(cfg, "CAPTION_BLEND", "convex")
        caption_shrink = bool(getattr(cfg, "CAPTION_SHRINK", False))
        caption_apply = getattr(cfg, "CAPTION_APPLY", "all")
        reliable_min = int(getattr(cfg, "CAPTION_RELIABLE_MIN", 2))
        global_mu = w_prompts_raw.mean(0)  # common "generic" direction, for CAPTION_CENTER
        # --- #2: how cap_w (caption/prompt agreement weight) is gated ---
        caption_gate = getattr(cfg, "CAPTION_GATE", "soft")   # soft (current) | hard (0/1) | freq (tail-scaled)
        gate_tau = float(getattr(cfg, "CAPTION_GATE_TAU", 0.0))
        _cn = torch.as_tensor(self.cls_num_list, dtype=torch.float32)
        _rank = torch.argsort(torch.argsort(_cn)).float()     # 0 = rarest ... N-1 = most frequent
        rarity = 1.0 - _rank / max(len(_cn) - 1, 1)           # rarest -> 1, most frequent -> 0
        few_set = set(torch.as_tensor(self.few_classes).flatten().tolist())
        headmed_set = (set(torch.as_tensor(self.many_classes).flatten().tolist())
                       | set(torch.as_tensor(self.med_classes).flatten().tolist()))

        all_caption_features = []
        for idx, _ in enumerate(tqdm(self.classnames, desc="Wiki caption encoding")):
            w_prompt_raw = w_prompts_raw[idx]

            sents = corpus.get(idx, [])
            if len(sents) == 0:
                if _dump is not None:
                    _dump.append({"idx": idx, "class": self.classnames[idx],
                                  "status": "no_corpus", "selected": []})
                all_caption_features.append(w_prompt_raw)
                continue
            
            """ 기존 truncate 방법 -> 뒷부분은 아예 잘려서 성능 하락 원인일수도
            sent_feats = []
            for i in range(0, len(sents), 128):
                batch = sents[i:i+128]
                tokens = clip.tokenize(batch, truncate=True).to(device)
                feats = F.normalize(self.model.text_encoder(tokens), dim=-1)
                sent_feats.append(feats)
            sent_feats = torch.cat(sent_feats, dim=0)
            """

            sent_feats_list = []
            
            # 128개씩 배치 처리하는 것은 유지 (메모리 관리)
            for i in range(0, len(sents), 128):
                batch_sents = sents[i:i+128] # 현재 배치(128개)의 문장들
                chunked_batch_sents = [] # 잘라낸 텍스트 조각들
                sent_indices = []        # 각 조각이 원래 몇 번째 문장 소속인지 (0~127)

                for sent_idx, sent_text in enumerate(batch_sents):
                    words = sent_text.split()
                    if not words:
                        continue # 빈 문장이면 건너뜀

                    if len(words) <= word_chunk_size:
                        # 40단어 이하면 그냥 통째로 사용
                        chunked_batch_sents.append(sent_text)
                        sent_indices.append(sent_idx)
                    else:
                        # 40단어 초과 시, 단어 단위로 잘라서 청크 생성
                        for j in range(0, len(words), word_chunk_size):
                            chunk_text = " ".join(words[j : j + word_chunk_size])
                            chunked_batch_sents.append(chunk_text)
                            sent_indices.append(sent_idx) # 원본 문장 인덱스 저장
                
                if not chunked_batch_sents:
                    continue # 이번 배치의 모든 문장이 비어있었음

                # 2. 모든 텍스트 조각(chunk)들을 한 번에 토큰화 및 인코딩
                tokens = clip.tokenize(chunked_batch_sents, truncate=True).to(device)
                chunk_feats = F.normalize(self.model.text_encoder(tokens), dim=-1) # [N_chunks, 512]
                
                # 3. 조각(chunk)들을 다시 원래 문장(sentence) 단위로 평균화
                sent_indices = torch.tensor(sent_indices, device=device)
                
                # [128, 512] 크기의 0 벡터 생성 (배치 크기만큼)
                batch_sent_feats = torch.zeros(len(batch_sents), chunk_feats.shape[1], 
                                               device=device, dtype=chunk_feats.dtype)
                
                # scatter_add_를 사용해 동일한 인덱스(문장)에 속한 조각(chunk) 벡터들을 모두 더함
                batch_sent_feats.scatter_add_(0, sent_indices.unsqueeze(1).expand_as(chunk_feats), chunk_feats)
                
                # 각 문장별로 몇 개의 조각이 더해졌는지 카운트
                counts = torch.zeros(len(batch_sents), device=device, dtype=torch.float32)
                counts.scatter_add_(0, sent_indices, torch.ones_like(sent_indices, dtype=torch.float32))
                counts = counts.clamp(min=1.0).unsqueeze(1) # 0으로 나누기 방지
                
                # 더해진 벡터를 카운트로 나누어 "평균 문장 벡터" 계산
                avg_feats = batch_sent_feats / counts
                
                # 평균화 후 다시 정규화
                avg_feats_norm = F.normalize(avg_feats, dim=-1)
                sent_feats_list.append(avg_feats_norm)
            
            if not sent_feats_list: # 클래스에 유효한 문장이 하나도 없었음
                if _dump is not None:
                    _dump.append({"idx": idx, "class": self.classnames[idx],
                                  "status": "no_valid_sents", "selected": []})
                all_caption_features.append(w_prompt_raw)
                continue
            
            sent_feats = torch.cat(sent_feats_list, dim=0) # [N_sents, 512]
            # 💡 --- >> 청킹 로직 끝 << --- 💡

            # 💡 --- >> 수정 5: 방법 2 (Hard Threshold) 적용 << --- 💡
            sims = sent_feats @ w_prompt_raw # [N_sents]
            k = min(top_k, sims.size(0))
            
            # top-k의 유사도 값과 인덱스를 모두 가져옴
            top_sims, top_idx = torch.topk(sims, k=k, largest=True)
            
            # Hard Threshold 적용: top-k 중에서도 sim_threshold를 넘는 것만 선택
            threshold_mask = top_sims > sim_threshold
            final_indices = top_idx[threshold_mask]
            
            selected = sent_feats[final_indices] # [N_selected, 512]

            # caption feature 평균
            if selected.shape[0] == 0:
                w_final = w_prompt_raw
                if _dump is not None:
                    # nothing passed the threshold: show the top-k that just missed
                    _dump.append({"idx": idx, "class": self.classnames[idx],
                                  "status": "below_threshold", "alpha": 1.0, "n_selected": 0,
                                  "top_missed": [{"sim": round(s, 4), "sent": sents[j]}
                                                 for s, j in zip(top_sims.tolist(), top_idx.tolist())]})
            else:
                cap_mean = selected.mean(0)
                if caption_center:                        # #3: remove the common "generic" direction
                    cap_mean = cap_mean - global_mu
                w_caption_raw = F.normalize(cap_mean, dim=-1)
                raw_trust_score = (w_prompt_raw * w_caption_raw).sum() # [-1, 1] 범위의 코사인 유사도
                trust_score = raw_trust_score.clamp(min=0.0).item()
                cap_w = trust_score                       # caption weight (= 1 - alpha)
                if caption_gate == "hard":                # #2: binary agreement gate
                    cap_w = 1.0 if trust_score > gate_tau else 0.0
                elif caption_gate == "freq":              # #2: scale caption by tail-ness (rarest -> full)
                    cap_w = trust_score * float(rarity[idx])
                n_sel = int(selected.shape[0])
                if caption_shrink:                        # #3: down-weight caption when few selected (tail noise)
                    cap_w = cap_w * n_sel / (n_sel + 1.0)
                if caption_blend == "residual":           # #3: add only caption comp. orthogonal to prompt
                    perp = w_caption_raw - (w_caption_raw * w_prompt_raw).sum() * w_prompt_raw
                    w_final = F.normalize(w_prompt_raw + cap_w * perp, dim=-1)
                else:                                     # convex (current default)
                    w_final = F.normalize((1.0 - cap_w) * w_prompt_raw + cap_w * w_caption_raw, dim=-1)
                if _dump is not None:
                    sel_sims = top_sims[threshold_mask].tolist()
                    _dump.append({"idx": idx, "class": self.classnames[idx], "status": "ok",
                                  "alpha": round(1.0 - cap_w, 4), "trust": round(trust_score, 4),
                                  "n_selected": n_sel,
                                  "selected": [{"sim": round(sm, 4), "sent": sents[j]}
                                               for j, sm in zip(final_indices.tolist(), sel_sims)]})

            # #4: does this class actually USE the caption blend? (else keep prompt-only)
            if caption_apply != "all":
                use_cap = ((caption_apply == "tail" and idx in few_set)
                           or (caption_apply == "headmed" and idx in headmed_set)
                           or (caption_apply == "reliable" and int(selected.shape[0]) >= reliable_min))
                if not use_cap:
                    w_final = w_prompt_raw

            all_caption_features.append(w_final)

        # 7️⃣ 최종 classifier weight로 사용
        self.class_features = torch.stack(all_caption_features, dim=0)
        print(f"[Wiki] Done: computed features for {len(self.classnames)} classes (top-{top_k}, thresh>{sim_threshold}, dynamic_alpha, chunked).")

        if _dump is not None:
            import json
            with open(_dump_path, "w", encoding="utf-8") as f:
                json.dump(_dump, f, ensure_ascii=False, indent=2)
            n_ok = sum(1 for d in _dump if d["status"] == "ok")
            print(f"[Wiki] Dumped caption selection to {_dump_path} "
                  f"({n_ok}/{len(_dump)} classes selected >=1 caption). Exiting (inspection mode).")
            sys.exit(0)

        return self.class_features
    
    def build_optimizer(self):
        cfg = self.cfg
        
        print("Turning off gradients in the model.")
        for param in self.model.parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner.")
        for param in self.tuner.parameters():
            param.requires_grad_(True)

        if getattr(cfg, "FREEZE_CLASSIFIER", False) and "classifier" in self.tuner:
            print("Freezing classifier: kept at init value, not trained.")
            for param in self.tuner["classifier"].parameters():
                param.requires_grad_(False)

        if getattr(cfg, "FREEZE_ENCODER", False):  # H_E: freeze PEFT/encoder, train ONLY the classifier
            assert not getattr(cfg, "FREEZE_CLASSIFIER", False), "FREEZE_ENCODER and FREEZE_CLASSIFIER can't both be set (nothing would train)."
            print("Freezing encoder (PEFT): only the classifier trains.")
            for name, param in self.tuner.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad_(False)

        self.optim = torch.optim.SGD(
            [p for p in self.tuner.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.optim.zero_grad()
        
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = torch.GradScaler("cuda") if cfg.prec_train == "amp" else None

    # -------------------------
    # Textual-prior regularizer
    # -------------------------
    def _get_clip_visual_proj(self):
        """Try to find CLIP ViT visual projection matrix (typically [D_img, D_txt])."""
        candidates = [
            ("image_proj",),
            ("visual", "proj"),
            ("model", "visual", "proj"),
            ("backbone", "visual", "proj"),
        ]
        for path in candidates:
            obj = self.model
            ok = True
            for key in path:
                if not hasattr(obj, key):
                    ok = False
                    break
                obj = getattr(obj, key)
            if ok and obj is not None:
                return obj
        return None

    @torch.no_grad()
    def _compute_text_prior_logits(self, image):
        """
        Teacher logits from a fixed textual prior (prompt/caption features):
            z = normalize(f_img)      (projected to text dim if needed)
            W_text = normalize(text_prior_weight)
            logits_text = s * (z @ W_text^T)
        """
        assert hasattr(self, "text_prior_weight"), "text_prior_weight not found; did you run classifier_init with text?"
        feat = self.model(image=image, return_feature=True)  # [B, D_img]
        W = self.text_prior_weight.to(device=feat.device)

        # dtype alignment
        target_dtype = feat.dtype
        W = W.to(dtype=target_dtype)

        # project image features if needed (e.g., ViT-B/16: 768 -> 512)
        if feat.size(-1) != W.size(-1):
            proj = self._get_clip_visual_proj()
            if proj is None:
                raise RuntimeError(
                    f"Feature dim mismatch: feat={feat.size(-1)} vs W_text={W.size(-1)}. "
                    "Could not find CLIP visual.proj to project features."
                )
            proj = proj.to(device=feat.device, dtype=target_dtype)
            feat = feat @ proj  # [B, D_txt]

        feat = F.normalize(feat, dim=-1)
        W = F.normalize(W, dim=-1)

        logits = feat @ W.t()
        # match cosine-classifier scale if used
        if hasattr(self.cfg, "classifier_scale") and self.cfg.classifier_scale is not None:
            logits = logits * float(self.cfg.classifier_scale)
        return logits

    def generate_class_prompts(self):
        prompts = [self.template.format(name.replace("_", " ")) for name in self.classnames]
        prompts = clip.tokenize(prompts)  # [n_cls, seq_len]
        prompts = prompts.to(self.device)
        return prompts

    def generate_class_prompts_with_template(self, template: str):
        prompts = [template.format(name.replace("_", " ")) for name in self.classnames]
        prompts = clip.tokenize(prompts)
        prompts = prompts.to(self.device)
        return prompts

    def compute_class_features(self, prompts):
        if len(prompts) <= 1000:
            class_features = self.model.text_encoder(prompts)
        else:
            # CUDA out of memory
            prompt_splits = torch.split(prompts, 1000)
            class_features = torch.cat([self.model.text_encoder(x) for x in prompt_splits])
        return class_features

    def compute_prompt_class_features(self):
        templates = self._get_prompt_templates()
        if len(templates) == 1:
            return self.compute_class_features(self.generate_class_prompts_with_template(templates[0]))

        class_features = []
        for template in templates:
            prompts = self.generate_class_prompts_with_template(template)
            features = self.compute_class_features(prompts)
            class_features.append(F.normalize(features, dim=-1))

        return F.normalize(torch.stack(class_features, dim=0).mean(dim=0), dim=-1)

    def _center_prototypes(self, feats):
        """#3: de-anisotropize prompt prototypes (subtract a centroid / whiten). feats: [C, D] unit rows."""
        mode = getattr(self.cfg, "PROMPT_CENTER_MODE", "global")
        orig_dtype = feats.dtype
        X = feats.float()
        if mode == "global":                     # subtract the global prompt centroid
            out = X - X.mean(0)
        elif mode == "group":                    # subtract the head(many)-group centroid
            head = torch.as_tensor(self.many_classes).flatten().to(X.device)
            out = X - X[head].mean(0)
        elif mode == "tail":                     # per-class strength ~ inverse frequency (rarest strongest)
            cn = torch.as_tensor(self.cls_num_list, dtype=torch.float32, device=X.device)
            rank = torch.argsort(torch.argsort(cn)).float()
            rarity = (1.0 - rank / max(len(cn) - 1, 1)).unsqueeze(1)   # [C,1] rarest->1, head->0
            out = X - rarity * X.mean(0)
        elif mode == "std":                      # diagonal whitening (standardize each dim)
            out = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
        elif mode == "whiten":                   # ZCA whitening (decorrelate + unit variance)
            Xc = X - X.mean(0)
            cov = (Xc.T @ Xc) / Xc.shape[0] + 1e-4 * torch.eye(X.shape[1], device=X.device)
            evals, evecs = torch.linalg.eigh(cov)
            W = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.T
            out = Xc @ W
        elif mode == "pca":                      # All-but-the-Top: mean-center + remove top-k principal comps
            k = int(getattr(self.cfg, "PROMPT_CENTER_PCA_K", 1))
            Xc = X - X.mean(0)                    # k=0 == global (mean only); k up to whiten-like collapse
            if k > 0:
                cov = (Xc.T @ Xc) / Xc.shape[0]
                _, evecs = torch.linalg.eigh(cov)  # ascending eigenvalues
                V = evecs[:, -k:]                  # top-k principal directions
                Xc = Xc - (Xc @ V) @ V.T           # project them out
            out = Xc
        # --- J negative controls: look like centering but do NOT remove the shared direction ---
        elif mode == "randdir":                  # subtract a RANDOM direction of matched norm (is it really mu?)
            gen = torch.Generator(device=X.device).manual_seed(int(getattr(self.cfg, "seed", 0)))
            u = F.normalize(torch.randn(X.shape[1], generator=gen, device=X.device), dim=0)
            out = X - X.mean(0).norm() * u
        elif mode == "headonly":                 # center head+med only, leave Few raw (must we touch tail?)
            mu = X.mean(0)
            sel = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
            sel[torch.as_tensor(self.many_classes).flatten().to(X.device)] = True
            sel[torch.as_tensor(self.med_classes).flatten().to(X.device)] = True
            out = X.clone()
            out[sel] = X[sel] - mu
        elif mode == "fewonly":                  # center Few only w/ GLOBAL mu; leave head+med raw (is tail the whole gain?)
            mu = X.mean(0)                        # mu computed over ALL classes, applied to tail alone
            sel = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
            sel[torch.as_tensor(self.few_classes).flatten().to(X.device)] = True
            out = X.clone()
            out[sel] = X[sel] - mu
        elif mode == "perclass_rand":            # independent random dir per class (pure init noise)
            gen = torch.Generator(device=X.device).manual_seed(int(getattr(self.cfg, "seed", 0)))
            U = F.normalize(torch.randn(X.shape, generator=gen, device=X.device), dim=-1)
            out = X - X.mean(0).norm() * U
        else:
            raise ValueError(f"unknown PROMPT_CENTER_MODE: {mode}")
        return F.normalize(out, dim=-1).to(orig_dtype)
    
    def compute_train_features(self):
        all_features = torch.Tensor([]).to(self.device)
        all_labels = torch.Tensor([]).to(self.device)

        print("Computing training features.")
        for image, label in tqdm(self.init_loader, ascii=True):
            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, return_feature=True)

            all_features = torch.cat([all_features, feature])
            all_labels = torch.cat([all_labels, label])

        return all_features, all_labels
        # sorted_index = all_labels.argsort()
        # all_features = all_features[sorted_index]
        # all_labels = all_labels[sorted_index]
        # _, label_counts = torch.unique(all_labels, return_counts=True)
        # class_means = torch.stack([x.mean(dim=0) for x in torch.split(all_features, label_counts.tolist())])
        # return class_means

    def _compute_text_prior_feat(self, image):
        """
        Return image features projected to text embedding dim (if needed), WITHOUT detaching.
        Used for feature-level InfoNCE so gradients can flow to the tuner.
        Output: [B, D_text]
        """
        feat = self.model(image=image, return_feature=True)  # [B, D_img]

        # text prior prototype shape for dim check
        W = self.text_prior_weight.to(device=feat.device, dtype=feat.dtype)  # [C, D_text]

        # project if needed (e.g., ViT-B/16: 768 -> 512)
        if feat.size(-1) != W.size(-1):
            proj = self._get_clip_visual_proj()
            if proj is None:
                raise RuntimeError(
                    f"Feature dim mismatch: feat={feat.size(-1)} vs W_text={W.size(-1)}. "
                    "Could not find CLIP visual.proj to project features."
                )
            proj = proj.to(device=feat.device, dtype=feat.dtype)
            feat = feat @ proj  # [B, D_text]

        return feat

    @torch.no_grad()
    def build_prior_gate(self):
        cfg = self.cfg
        source = getattr(cfg, "PRIOR_GATE_SOURCE", "image_text")
        sim, valid = self._compute_gate_signal(source)

        # The raw signal (e.g. cosine) has a tiny, geometry-dependent dynamic range
        # that would silently nullify the gated loss. What matters is the *relative*
        # trust across classes, so rescale to a usable [0, 1] over valid classes.
        norm_mode = getattr(cfg, "PRIOR_GATE_NORM", "minmax")
        gate = self._normalize_gate(sim, valid, norm_mode)
        invert = bool(getattr(cfg, "PRIOR_GATE_INVERT", False))
        if invert:
            # flip direction: low-signal classes get the stronger prior.
            gate = torch.where(valid, 1.0 - gate, gate)
        gate = gate.pow(float(getattr(cfg, "PRIOR_GATE_POWER", 1.0)))
        gate = torch.where(valid, gate, torch.zeros_like(gate))

        if source == "shuffled":
            # Negative control: keep the exact gate VALUE distribution but destroy
            # the class<->weight mapping. If the agreement SIGNAL (not merely having
            # a non-uniform gate) is what matters, this should NOT recover the gain.
            gate = self._shuffle_gate(gate, valid)

        self.prior_gate = gate.detach()
        sv, gv = sim[valid], gate[valid]
        print(
            "[PriorGate] source={} signal: min={:.4f} mean={:.4f} max={:.4f} | "
            "norm={} invert={} gate: min={:.4f} mean={:.4f} max={:.4f}".format(
                source, sv.min().item(), sv.mean().item(), sv.max().item(),
                norm_mode, invert, gv.min().item(), gv.mean().item(), gv.max().item(),
            )
        )

    def _compute_gate_signal(self, source):
        """Return (raw per-class signal, valid mask) that drives the class gate."""
        if source in ("image_text", "shuffled"):
            if not hasattr(self, "text_prior_weight"):
                raise RuntimeError(
                    "text_prior_weight is required for PRIOR_GATE_SOURCE=image_text/shuffled.")
            print(f"[PriorGate] signal = cos(train image-mean, text prior)  [source={source}]")
            text_prior = F.normalize(self.text_prior_weight.float(), dim=-1)
            feat_dim = text_prior.size(1)
            sums = torch.zeros(self.num_classes, feat_dim, device=self.device, dtype=torch.float32)
            counts = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)
            for image, label in tqdm(self.init_loader, ascii=True, desc="Prior gate"):
                image = image.to(self.device)
                label = label.to(self.device)
                feat = F.normalize(self._compute_text_prior_feat(image).float(), dim=-1)
                sums.index_add_(0, label, feat)
                counts.index_add_(0, label, torch.ones_like(label, dtype=torch.float32))
            means = F.normalize(sums / counts.clamp(min=1.0).unsqueeze(1), dim=-1)
            sim = (means * text_prior).sum(dim=1)  # raw per-class cosine
            return sim, counts > 0

        if source == "frequency":
            # Alternative axis / control: gate by class train frequency, NOT
            # image-text agreement. Tests "is the agreement gate just frequency?"
            print("[PriorGate] signal = per-class train frequency")
            sim = torch.tensor(self.cls_num_list, device=self.device, dtype=torch.float32)
            return sim, sim > 0

        raise ValueError(f"Unknown PRIOR_GATE_SOURCE: {source}")

    def _shuffle_gate(self, gate, valid):
        """Permute gate values among valid classes (deterministic w.r.t. cfg.seed)."""
        seed = int(getattr(self.cfg, "seed", 0) or 0)
        g = torch.Generator().manual_seed(seed)
        gate_cpu = gate.detach().cpu()
        idx = torch.where(valid.cpu())[0]
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        out = gate_cpu.clone()
        out[idx] = gate_cpu[perm]
        return out.to(gate.device)

    @staticmethod
    def _normalize_gate(sim, valid, mode):
        """Map raw per-class similarity to a [0, 1] gate over the valid classes."""
        gate = torch.zeros_like(sim)
        if valid.sum() == 0:
            return gate
        sv = sim[valid]
        if mode == "none":
            gate[valid] = sv.clamp(min=0.0, max=1.0)
        elif mode == "minmax":
            lo, hi = sv.min(), sv.max()
            gate[valid] = ((sv - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)
        elif mode == "rank":
            # rank in [0, 1]: lowest-similarity class -> 0, highest -> 1
            order = torch.argsort(torch.argsort(sv)).float()
            denom = max(sv.numel() - 1, 1)
            gate[valid] = order / denom
        else:
            raise ValueError(f"Unknown PRIOR_GATE_NORM: {mode}")
        return gate

    def _reg_anneal_scale(self, epoch_idx, num_epochs):
        """Scale factor in [REG_ANNEAL_END, 1.0] applied to the KD/InfoNCE lambdas.

        Strong (1.0) early to stabilize the text-prior-based init, decaying to
        REG_ANNEAL_END late so LA can fit the classifier to the visual boundary.
        """
        cfg = self.cfg
        mode = getattr(cfg, "REG_ANNEAL", "none")
        if mode == "none":
            return 1.0
        end = float(getattr(cfg, "REG_ANNEAL_END", 0.0))
        start = int(getattr(cfg, "REG_ANNEAL_START_EPOCH", 0))
        if epoch_idx < start:
            return 1.0
        denom = max(num_epochs - 1 - start, 1)
        p = (epoch_idx - start) / denom  # 0 -> 1 across the decay window
        if mode == "linear":
            return 1.0 - (1.0 - end) * p
        if mode == "cosine":
            return end + (1.0 - end) * 0.5 * (1.0 + math.cos(math.pi * p))
        raise ValueError(f"Unknown REG_ANNEAL: {mode}")

    def apply_prior_gate(self, per_sample_loss, label):
        if getattr(self.cfg, "PRIOR_REG_MODE", "fixed") == "fixed":
            return per_sample_loss.mean()
        if getattr(self.cfg, "PRIOR_REG_MODE", "fixed") != "class_gate":
            raise ValueError(f"Unknown PRIOR_REG_MODE: {self.cfg.PRIOR_REG_MODE}")

        if not hasattr(self, "prior_gate"):
            self.build_prior_gate()

        weights = self.prior_gate.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)[label]
        return (weights * per_sample_loss).mean()

    # ------------------------------------------------------------------
    # Warmup helper
    # ------------------------------------------------------------------
    def _warmup_select_params(self):
        """Return warmup-eligible parameters."""
        cfg = self.cfg

        warm_image = bool(getattr(cfg, "PEFT_WARMUP_IMAGE", True))
        warm_text  = bool(getattr(cfg, "PEFT_WARMUP_TEXT", False))

        params = []

        if warm_image and ("image_encoder" in self.tuner):
            params += list(self.tuner["image_encoder"].parameters())

        if warm_text and ("text_encoder" in self.tuner):
            params += list(self.tuner["text_encoder"].parameters())

        # (선택) projection까지 warmup하고 싶으면 켜기
        if bool(getattr(cfg, "PEFT_WARMUP_PROJ", False)):
            if "image_proj" in self.tuner:
                params += list(self.tuner["image_proj"].parameters())
            if "text_proj" in self.tuner:
                params += list(self.tuner["text_proj"].parameters())

        # classifier는 warm-start에서 기본적으로 제외 (원하면 stage1에서 학습)
        if bool(getattr(cfg, "PEFT_WARMUP_CLASSIFIER", False)) and ("classifier" in self.tuner):
            params += list(self.tuner["classifier"].parameters())

        # 중복 제거
        seen = set()
        uniq = []
        for p in params:
            if id(p) not in seen:
                uniq.append(p)
                seen.add(id(p))
        return uniq

    def warmup_peft(self):
        """Stage 0: warm up PEFT modules with KD + InfoNCE, without CE.
        After warmup, rebuilds optimizer for normal training.
        """
        cfg = self.cfg

        # 이미 했으면 스킵
        if getattr(self, "_peft_warmup_done", False):
            return

        # enable flag
        if not bool(getattr(cfg, "PEFT_WARMUP", False)):
            return

        # text prior 필요 (KD/InfoNCE 모두 text_prior_weight를 씀)
        if not hasattr(self, "text_prior_weight"):
            print("⚠️ [Warmup] text_prior_weight not found. "
                  "Set classifier_init=semantic/hybrid to build it. Skip warmup.")
            self._peft_warmup_done = True
            return

        # warmup step/epoch 설정
        warm_epochs = int(getattr(cfg, "PEFT_WARMUP_EPOCHS", 1))
        warm_steps  = int(getattr(cfg, "PEFT_WARMUP_STEPS", -1))  # >0이면 steps 우선
        warm_lr     = float(getattr(cfg, "PEFT_WARMUP_LR", cfg.lr))

        # warmup에서 쓸 KD/InfoNCE 하이퍼 (없으면 stage1 값 재사용)
        text_reg_lambda = float(getattr(cfg, "WARMUP_TEXT_REG_LAMBDA", getattr(cfg, "TEXT_REG_LAMBDA", 0.0)))
        text_reg_T      = float(getattr(cfg, "WARMUP_TEXT_REG_T",      getattr(cfg, "TEXT_REG_T", 1.0)))
        infonce_lambda  = float(getattr(cfg, "WARMUP_INFONCE_LAMBDA",   getattr(cfg, "INFONCE_LAMBDA", 0.0)))
        infonce_T       = float(getattr(cfg, "WARMUP_INFONCE_T",        getattr(cfg, "INFONCE_T", 0.1)))

        if (text_reg_lambda <= 0) and (infonce_lambda <= 0):
            print("⚠️ [Warmup] Both KD/InfoNCE lambdas are 0. Skip warmup.")
            self._peft_warmup_done = True
            return

        # KD는 student logits가 필요 -> classifier 모드에서만 가능
        if text_reg_lambda > 0 and (not cfg.classifier):
            print("⚠️ [Warmup] KD enabled but cfg.classifier=False. Disable KD for warmup.")
            text_reg_lambda = 0.0

        # loss modules 준비
        warm_kd_loss = None
        warm_nce_loss = None
        if text_reg_lambda > 0:
            warm_kd_loss = LogitKDLoss(T=text_reg_T)
            print(f"→ [Warmup] KD enabled: lambda={text_reg_lambda}, T={text_reg_T}")
        if infonce_lambda > 0:
            warm_nce_loss = InfoNCELoss(T=infonce_T, reduction="mean")
            print(f"→ [Warmup] InfoNCE enabled: lambda={infonce_lambda}, T={infonce_T}")

        # -------------------------
        # requires_grad 제어: 모두 off 후 warmup params만 on
        # -------------------------
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.tuner.parameters():
            p.requires_grad_(False)

        # model forward args
        if cfg.classifier:
            model_args = {"use_classifier": True}
        else:
            with torch.no_grad():
                text = self.compute_prompt_class_features()
            model_args = {"text": text, "is_text_feature": True}

        self.tuner.train()
        scaler = self.scaler if cfg.prec_train == "amp" else None
        num_batches = len(self.train_loader)

        def _run_one_step(image, label, optim_w, step, total_steps):
            """Run a single warmup gradient step. Returns scalar loss."""
            def _compute_loss():
                logit = None
                if warm_kd_loss is not None:
                    logit = self.model(image=image, **model_args)

                loss = 0.0

                if warm_kd_loss is not None:
                    with torch.no_grad():
                        text_logit = self._compute_text_prior_logits(image)
                    loss = loss + text_reg_lambda * warm_kd_loss(logit, text_logit)

                if warm_nce_loss is not None:
                    feat_txt = self._compute_text_prior_feat(image)
                    nce = warm_nce_loss(feat_txt, self.text_prior_weight, label)
                    loss = loss + infonce_lambda * nce

                return loss

            if cfg.prec_train == "amp":
                with torch.autocast(device_type="cuda"):
                    loss = _compute_loss()
                scaler.scale(loss / cfg.accum_step).backward()
                if (step + 1) % cfg.accum_step == 0:
                    scaler.step(optim_w)
                    scaler.update()
                    optim_w.zero_grad()
            else:
                loss = _compute_loss()
                (loss / cfg.accum_step).backward()
                if (step + 1) % cfg.accum_step == 0:
                    optim_w.step()
                    optim_w.zero_grad()

            return float(loss.detach())

        warm_params = self._warmup_select_params()
        if len(warm_params) == 0:
            print("⚠️ [Warmup] No warmup params selected. Skip.")
            self._peft_warmup_done = True
            return

        for p in warm_params:
            p.requires_grad_(True)

        optim_w = torch.optim.AdamW(warm_params, lr=warm_lr,
                                    weight_decay=cfg.weight_decay)

        total_steps = warm_steps if warm_steps > 0 else warm_epochs * num_batches
        step = 0

        print("==============================================================")
        print(f"[Warmup] Start: lr={warm_lr}, epochs={warm_epochs}, steps={warm_steps}, "
              f"params={sum(p.numel() for p in warm_params)}")
        print("==============================================================")

        for epoch in range(10**9):
            for batch_idx, (image, label) in enumerate(self.train_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                loss_val = _run_one_step(image, label, optim_w, step, total_steps)
                if (step + 1) % cfg.print_freq == 0 or step == 0:
                    print(f"[Warmup] step {step+1}/{total_steps} loss={loss_val:.4f}")
                    sys.stdout.flush()
                step += 1
                if step >= total_steps:
                    break
            if step >= total_steps:
                break

        print("[Warmup] Done. Rebuild optimizer for normal training.")

        # Release warmup optimizer (AdamW m/v states) before building the new one
        del optim_w
        if warm_kd_loss is not None:
            del warm_kd_loss
        if warm_nce_loss is not None:
            del warm_nce_loss
        torch.cuda.empty_cache()

        # Save warmup-end checkpoint for visualization
        warm_dir = os.path.join(cfg.output_dir, "ckpts", "after_warmup")
        os.makedirs(warm_dir, exist_ok=True)
        self.save_model(warm_dir)

        # Stage1을 위해 원래 로직 복구: tuner 전체 학습 + SGD/cosine 등
        self.build_optimizer()

        self._peft_warmup_done = True

    def train(self):
        cfg = self.cfg

        # Save initial (pre-training) checkpoint for visualization
        init_dir = os.path.join(cfg.output_dir, "ckpts", "init")
        os.makedirs(init_dir, exist_ok=True)
        self.save_model(init_dir)

        # ---- PEFT warm-start (Stage 0) ----
        if bool(getattr(cfg, "PEFT_WARMUP", False)) and (not getattr(self, "_peft_warmup_done", False)):
            self.warmup_peft()
        # -----------------------------------

        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        tb_writer = SummaryWriter(log_dir=writer_dir)

        batch_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        if cfg.classifier:
            model_args = {"use_classifier": True}
        else:
            print("Pre-computing class prompt features.")
            with torch.no_grad():
                text = self.compute_prompt_class_features()
            model_args = {"text": text, "is_text_feature": True}

        # Textual Prior (KL)
        self.text_reg_lambda = float(getattr(cfg, "TEXT_REG_LAMBDA", 0.0))
        self.text_reg_T = float(getattr(cfg, "TEXT_REG_T", 1.0))

        # InfoNCE
        self.infonce_lambda = float(getattr(cfg, "INFONCE_LAMBDA", 0.0))
        self.infonce_T = float(getattr(cfg, "INFONCE_T", 0.1))

        # Validate text prior availability
        if (self.text_reg_lambda > 0) or (self.infonce_lambda > 0):
            if not hasattr(self, "text_prior_weight"):
                print("⚠️ Text prior losses enabled but no text_prior_weight found. Disable KD/InfoNCE.")
                self.text_reg_lambda = 0.0
                self.infonce_lambda = 0.0

        # Build loss modules
        if self.text_reg_lambda > 0:
            if not hasattr(self, "text_reg_loss"):
                self.text_reg_loss = LogitKDLoss(T=self.text_reg_T)
            else:
                self.text_reg_loss.T = self.text_reg_T
            print(f"→ Using textual prior KL(KD): lambda={self.text_reg_lambda}, T={self.text_reg_T}")

        if self.infonce_lambda > 0:
            if not hasattr(self, "infonce_loss"):
                self.infonce_loss = InfoNCELoss(T=self.infonce_T, reduction="mean")
            else:
                self.infonce_loss.T = self.infonce_T
            print(f"→ Using feature-level InfoNCE: lambda={self.infonce_lambda}, T={self.infonce_T}")

        print("Start training")
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            # KD/InfoNCE annealing: scale the reg lambdas for this epoch.
            reg_scale = self._reg_anneal_scale(epoch_idx, num_epochs)
            if getattr(cfg, "REG_ANNEAL", "none") != "none":
                print(f"[RegAnneal] epoch {epoch_idx + 1}/{num_epochs} reg_scale={reg_scale:.4f}")

            num_batches = len(self.train_loader)
            for batch_idx, (image, label) in enumerate(self.train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                kd_loss = None
                nce_loss = None

                if cfg.prec_train == "amp":
                    with torch.autocast(device_type="cuda"):
                        logit = self.model(image=image, **model_args)
                        ce_loss = self.criterion(logit, label)

                        loss = ce_loss

                        # (A) textual prior KL (teacher logits)
                        if self.text_reg_lambda > 0:
                            with torch.no_grad():
                                text_logit = self._compute_text_prior_logits(image)
                            if getattr(cfg, "PRIOR_REG_MODE", "fixed") == "class_gate":
                                kd_per_sample = self.text_reg_loss(logit, text_logit, reduction="none")
                                kd_loss = self.apply_prior_gate(kd_per_sample, label)
                            else:
                                kd_loss = self.text_reg_loss(logit, text_logit)
                            loss = loss + (self.text_reg_lambda * reg_scale) * kd_loss

                        # (B) feature-level InfoNCE vs fixed text prototypes
                        if self.infonce_lambda > 0:
                            # IMPORTANT: this keeps grad (no no_grad)
                            feat_txt = self._compute_text_prior_feat(image)  # [B, D_text]
                            if getattr(cfg, "PRIOR_REG_MODE", "fixed") == "class_gate":
                                nce_per_sample = self.infonce_loss(feat_txt, self.text_prior_weight, label, reduction="none")
                                nce_loss = self.apply_prior_gate(nce_per_sample, label)
                            else:
                                nce_loss = self.infonce_loss(feat_txt, self.text_prior_weight, label)
                            loss = loss + (self.infonce_lambda * reg_scale) * nce_loss

                    self.scaler.scale(loss / cfg.accum_step).backward()
                    if ((batch_idx + 1) % cfg.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()

                else:
                    logit = self.model(image=image, **model_args)
                    ce_loss = self.criterion(logit, label)

                    loss = ce_loss

                    if self.text_reg_lambda > 0:
                        with torch.no_grad():
                            text_logit = self._compute_text_prior_logits(image)
                        if getattr(cfg, "PRIOR_REG_MODE", "fixed") == "class_gate":
                            kd_per_sample = self.text_reg_loss(logit, text_logit, reduction="none")
                            kd_loss = self.apply_prior_gate(kd_per_sample, label)
                        else:
                            kd_loss = self.text_reg_loss(logit, text_logit)
                        loss = loss + (self.text_reg_lambda * reg_scale) * kd_loss

                    if self.infonce_lambda > 0:
                        feat_txt = self._compute_text_prior_feat(image)
                        if getattr(cfg, "PRIOR_REG_MODE", "fixed") == "class_gate":
                            nce_per_sample = self.infonce_loss(feat_txt, self.text_prior_weight, label, reduction="none")
                            nce_loss = self.apply_prior_gate(nce_per_sample, label)
                        else:
                            nce_loss = self.infonce_loss(feat_txt, self.text_prior_weight, label)
                        loss = loss + (self.infonce_lambda * reg_scale) * nce_loss

                    (loss / cfg.accum_step).backward()
                    if ((batch_idx + 1) % cfg.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                # metrics
                with torch.no_grad():
                    pred = logit.argmax(dim=1)
                    correct = pred.eq(label)
                    acc = correct.float().mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.float().mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = torch.mean(torch.Tensor(cls_accs))
                many_acc = torch.mean(torch.Tensor(cls_accs)[self.many_classes])
                med_acc  = torch.mean(torch.Tensor(cls_accs)[self.med_classes])
                few_acc  = torch.mean(torch.Tensor(cls_accs)[self.few_classes])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = (num_batches - batch_idx - 1) + (num_epochs - epoch_idx - 1) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})"]
                    info += [f"(mean {mean_acc:.2f} many {many_acc:.2f} med {med_acc:.2f} few {few_acc:.2f})"]
                    if kd_loss is not None:
                        info += [f"kd {kd_loss.item():.4f}"]
                    if nce_loss is not None:
                        info += [f"nce {nce_loss.item():.4f}"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))
                    sys.stdout.flush()

                iter_idx = epoch_idx * num_batches + batch_idx
                tb_writer.add_scalar("train/lr", current_lr, iter_idx)
                tb_writer.add_scalar("train/loss.val", loss_meter.val, iter_idx)
                tb_writer.add_scalar("train/loss.avg", loss_meter.avg, iter_idx)
                tb_writer.add_scalar("train/acc.val", acc_meter.val, iter_idx)
                tb_writer.add_scalar("train/acc.avg", acc_meter.avg, iter_idx)
                tb_writer.add_scalar("train/mean_acc", mean_acc, iter_idx)
                tb_writer.add_scalar("train/many_acc", many_acc, iter_idx)
                tb_writer.add_scalar("train/med_acc", med_acc, iter_idx)
                tb_writer.add_scalar("train/few_acc", few_acc, iter_idx)
                tb_writer.add_scalar("train/ce_loss", ce_loss.item(), iter_idx)
                if kd_loss is not None:
                    tb_writer.add_scalar("train/kd_loss", kd_loss.item(), iter_idx)
                if nce_loss is not None:
                    tb_writer.add_scalar("train/nce_loss", nce_loss.item(), iter_idx)

                end = time.time()

            self.sched.step()
            for t in self.train_loader.dataset.transform.transforms:
                if isinstance(t, MinimalistRandomResizedCrop):
                    t.step()

        print("Finish training")
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        self.save_model(cfg.output_dir)
        tb_writer.close()

    def test(self):
        cfg = self.cfg
        self.tuner.eval()

        prec = cfg.prec_test
        if prec == "fp16":
            self.model.half()
        elif prec == "fp32":
            self.model.float()
        else:
            raise ValueError

        print(f"Evaluate on the test set")

        evaluator = Evaluator()

        if cfg.classifier:
            if getattr(cfg, "EVAL_CENTER", False):  # H_B: decision-time centering of the TRAINED classifier
                with torch.no_grad():
                    W = self.tuner["classifier"].weight.data
                    self.tuner["classifier"].weight.data = self._center_prototypes(F.normalize(W, dim=1)).to(W.dtype)
                print(f"[EVAL_CENTER] centered TRAINED classifier weight before eval "
                      f"(mode={getattr(cfg, 'PROMPT_CENTER_MODE', 'global')}).")
            model_args = {"use_classifier": True}
        else:
            print("Pre-computing class features for testing.")
            with torch.no_grad():
                text = self.compute_prompt_class_features()
                if getattr(cfg, "PROMPT_CENTER", False):  # exp1: zero-shot centering (no classifier, no training)
                    text = self._center_prototypes(text)
                    print(f"[PROMPT_CENTER] zero-shot mode={getattr(cfg, 'PROMPT_CENTER_MODE', 'global')} "
                          f"applied to text prototypes.")
            model_args = {"text": text, "is_text_feature": True}
        
        save_logits = bool(getattr(cfg, "SAVE_LOGITS", False))
        all_logits = [] if save_logits else None

        for image, label in tqdm(self.test_loader, ascii=True, desc="Testing"):
            image = image.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if cfg.tte:  # [bsz, ncrops, C, H, W]
                    logit = torch.stack([self.model(image=x, **model_args) for x in image.unbind(dim=1)]).mean(dim=0)
                else:
                    logit = self.model(image=image, **model_args)

            evaluator.process(logit, label)
            if all_logits is not None:
                all_logits.append(logit.float().cpu())

        cls_accs = evaluator.evaluate(self.many_classes, self.med_classes, self.few_classes)

        if all_logits is not None:
            import numpy as np
            np.save(os.path.join(cfg.output_dir, "logits.npy"),
                    torch.cat(all_logits).half().numpy())

        # Save per-class accuracy for visualization
        import numpy as np
        np.save(os.path.join(cfg.output_dir, "cls_accs.npy"), cls_accs.numpy())
        np.save(os.path.join(cfg.output_dir, "cls_num_list.npy"), np.asarray(self.cls_num_list))
        # Dump raw predictions/labels so confusion (e.g. which shot-group med leaks into) can be analyzed offline.
        np.save(os.path.join(cfg.output_dir, "y_true.npy"), np.asarray(evaluator._y_true))
        np.save(os.path.join(cfg.output_dir, "y_pred.npy"), np.asarray(evaluator._y_pred))
        if hasattr(self, "prior_gate"):
            np.save(os.path.join(cfg.output_dir, "prior_gate.npy"), self.prior_gate.float().cpu().numpy())

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)
        tuner_dict = checkpoint["tuner"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)
