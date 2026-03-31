import os
import sys
import time
import datetime
import math

from collections import OrderedDict
from contextlib import contextmanager
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

        # 💡 --- >> 수정된 부분 시작 << --- 💡
        # 각 클래스 인덱스별로 Alpha를 빠르게 조회하기 위한 불리언 마스크를 생성합니다.
        print("Creating class-specific masks for dynamic alpha...")
        self.many_mask = (torch.tensor(self.cls_num_list) > 100)
        self.med_mask = ((torch.tensor(self.cls_num_list) >= 20) & (torch.tensor(self.cls_num_list) <= 100))
        self.few_mask = (torch.tensor(self.cls_num_list) < 20)
        # 💡 --- >> 수정된 부분 끝 << --- 💡

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
    
    def build_wiki_corpus(
        self,
        caption_dir: str,
        classnames: List[str],
        max_sentences: int = 0,
        max_chars: int = 0
    ) -> Dict[int, List[str]]:
        corpus = {}
        for i, _ in enumerate(classnames):
            caption_path = os.path.join(caption_dir, f"desc_{i}.txt")
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
            self.template = "a photo of a {}."
            print("Use template '{}' for prompt generation.".format(self.template))

        def _scale_dim(base_dim: int, scale: float, keep_pow2: bool, max_dim: int):
            x = max(1, int(round(base_dim * float(scale))))
            if keep_pow2:
                x = 2 ** max(0, int(round(math.log2(x))))
            return min(x, max_dim)

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

            mix_mode = getattr(_cfg, "hybrid_mix_mode", "parallel")
            head_tail_order = getattr(_cfg, "hybrid_head_tail_order", "lora_first")
            lora_layers = parse_layers(_cfg.lora_layers) if _cfg.lora else []
            adaptformer_layers = parse_layers(_cfg.adaptformer_layers) if _cfg.adaptformer else []
            if (_name == "image_encoder") and mix_mode == "head_tail" and _cfg.lora and _cfg.adaptformer:
                n_layers = len(_encoder.blocks)
                head_count = int(getattr(_cfg, "hybrid_head_layers", 0) or max(1, n_layers // 2))
                tail_count = int(getattr(_cfg, "hybrid_tail_layers", 0) or max(1, n_layers - head_count))
                head_count = min(head_count, n_layers)
                tail_start = max(head_count, n_layers - min(tail_count, n_layers))
                if head_tail_order == "af_first":
                    # 초기 layer: AdaptFormer (범용 표현), 후반 layer: rank-scaled LoRA (세밀 조정)
                    adaptformer_layers = list(range(head_count))
                    lora_layers = list(range(tail_start, n_layers))
                    print(f"[HybridMix] head_tail af_first: AdaptFormer={adaptformer_layers}, LoRA={lora_layers}")
                else:
                    # lora_first (default): 초기=LoRA, 후반=AdaptFormer
                    lora_layers = list(range(head_count))
                    adaptformer_layers = list(range(tail_start, n_layers))
                    print(f"[HybridMix] head_tail lora_first: LoRA={lora_layers}, AdaptFormer={adaptformer_layers}")
            elif (_name == "image_encoder") and mix_mode == "gated_parallel" and _cfg.lora and _cfg.adaptformer:
                # 동일 layer에 LoRA(attention) + AdaptFormer + FFN-LoRA 모두 적용
                # → h = Wx + α·FFN_LoRA(x) + β·AdaptFormer(x)  (+ attention LoRA)
                lora_layers = parse_layers(_cfg.lora_layers)
                adaptformer_layers = lora_layers
                print(f"[HybridMix] gated_parallel: LoRA+AdaptFormer on layers {lora_layers}")

            if _cfg.lora and len(lora_layers) > 0:
                base_dim = _cfg.lora_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(lora_layers) * 4))))
                bottle_dim = _scale_dim(
                    base_dim,
                    getattr(_cfg, "lora_dim_scale", 1.0),
                    getattr(_cfg, "keep_bottleneck_pow2", True),
                    _encoder.embed_dim,
                )
                print(f"Add LoRA with bottle dimension {bottle_dim} (base={base_dim}) to layers {lora_layers}.")
                _encoder.add_lora(
                    layers=lora_layers,
                    bottle_dim=bottle_dim,
                    scale_init=getattr(_cfg, "lora_gate_scale", 1.0),
                    learnable_scale=bool(getattr(_cfg, "lora_gate_learnable", False)),
                    scale_q=getattr(_cfg, "lora_gate_scale_q", None),
                    scale_v=getattr(_cfg, "lora_gate_scale_v", None),
                )

                if getattr(_cfg, "lora_layers_last", None) is not None:
                    layers_last = [i for i in parse_layers(_cfg.lora_layers_last) if i in lora_layers]
                    if len(layers_last) > 0:
                        base_last = _cfg.lora_dim_last or base_dim
                        bottle_last = _scale_dim(
                            base_last,
                            getattr(_cfg, "lora_dim_last_scale", 1.0),
                            getattr(_cfg, "keep_bottleneck_pow2", True),
                            _encoder.embed_dim,
                        )
                        print(f"Override LoRA dim {bottle_last} on last layers {layers_last}.")
                        _encoder.add_lora(
                            layers=layers_last,
                            bottle_dim=bottle_last,
                            scale_init=getattr(_cfg, "lora_gate_scale", 1.0),
                            learnable_scale=bool(getattr(_cfg, "lora_gate_learnable", False)),
                            scale_q=getattr(_cfg, "lora_gate_scale_q", None),
                            scale_v=getattr(_cfg, "lora_gate_scale_v", None),
                        )

            if _cfg.adapter:
                layers = parse_layers(_cfg.adapter_layers)
                bottle_dim = _cfg.adapter_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(layers) * 2))))
                print("Add Adapter with bottle dimension {} to layers {}.".format(bottle_dim, layers))
                _encoder.add_adapter(layers=layers, bottle_dim=bottle_dim)

            if _cfg.adaptformer and len(adaptformer_layers) > 0:
                def _round_pow2(x: int) -> int:
                    x = max(1, int(x))
                    return 2 ** max(0, int(round(math.log2(x))))

                base_dim = _cfg.adaptformer_dim or 2 ** max(
                    0, int(math.log2(self.num_classes / (len(adaptformer_layers) * 2)))
                )
                dim_scale = float(getattr(_cfg, "adaptformer_dim_scale", 1.0))
                bottle_dim = _round_pow2(base_dim * dim_scale)
                print(f"[AdaptFormer] base layers={adaptformer_layers}, base_dim={base_dim}, "
                      f"scale={dim_scale} -> bottle_dim={bottle_dim}")
                _encoder.add_adaptformer(
                    layers=adaptformer_layers,
                    bottle_dim=bottle_dim,
                    scale_init=getattr(_cfg, "adaptformer_gate_scale", 1.0),
                    learnable_scale=bool(getattr(_cfg, "adaptformer_gate_learnable", True)),
                )

                layers_last_cfg = getattr(_cfg, "adaptformer_layers_last", None)
                if layers_last_cfg is not None:
                    layers_last = [i for i in parse_layers(layers_last_cfg) if i in adaptformer_layers]
                    if len(layers_last) > 0:
                        base_last = getattr(_cfg, "adaptformer_dim_last", None)
                        if base_last is None:
                            base_last = base_dim
                        last_scale = float(getattr(_cfg, "adaptformer_dim_last_scale", 1.0))
                        bottle_last = _round_pow2(base_last * last_scale)
                        print(f"[AdaptFormer] OVERRIDE last layers={layers_last}, "
                              f"base_last={base_last}, last_scale={last_scale} -> bottle_last={bottle_last}")
                        _encoder.add_adaptformer(
                            layers=layers_last,
                            bottle_dim=bottle_last,
                            scale_init=getattr(_cfg, "adaptformer_gate_scale", 1.0),
                            learnable_scale=bool(getattr(_cfg, "adaptformer_gate_learnable", True)),
                        )
            
            # FFN-LoRA: block-level LoRA bypass (h = Wx + α·LoRA(x) + β·AdaptFormer(x))
            # Activated by: v.ffn_lora=True (standalone) or mix_mode=gated_parallel (auto)
            ffn_lora_enabled = getattr(_cfg, "ffn_lora", False)
            is_gated_parallel = (_name == "image_encoder") and mix_mode == "gated_parallel" and _cfg.lora and _cfg.adaptformer
            if ffn_lora_enabled or is_gated_parallel:
                ffn_lora_layers = lora_layers if (is_gated_parallel or _cfg.lora) else adaptformer_layers
                if len(ffn_lora_layers) > 0:
                    base_dim_ffn = getattr(_cfg, "ffn_lora_dim", None) or _cfg.lora_dim or \
                        2 ** max(0, int(math.log2(self.num_classes / (len(ffn_lora_layers) * 4))))
                    bottle_dim_ffn = _scale_dim(base_dim_ffn, 1.0, getattr(_cfg, "keep_bottleneck_pow2", True), _encoder.embed_dim)
                    print(f"[FFN-LoRA] block-level LoRA bypass dim={bottle_dim_ffn}, layers={ffn_lora_layers}")
                    _encoder.add_ffn_lora(
                        layers=ffn_lora_layers,
                        bottle_dim=bottle_dim_ffn,
                        scale_init=float(getattr(_cfg, "ffn_lora_gate_scale", 1.0)),
                        learnable_scale=bool(getattr(_cfg, "ffn_lora_gate_learnable", True)),
                    )

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

            if not (cfg.zero_shot or cfg.test_only) and cfg.classifier_init is not None:
                classifier_init = cfg.classifier_init
                
                if classifier_init == "semantic":
                    print("Using semantic-aware initialization.")
                    with torch.no_grad():
                        class_features = self.compute_class_features(self.generate_class_prompts())
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
        )

        print(f"[Wiki] Computing features (top-{top_k}, thresh>{sim_threshold}, dynamic_alpha, chunked) for dataset={cfg.dataset} ...")

        prompts = self.generate_class_prompts()
        w_prompts_raw = self.compute_class_features(prompts)
        w_prompts_raw = F.normalize(w_prompts_raw, dim=-1)

        all_caption_features = []
        for idx, _ in enumerate(tqdm(self.classnames, desc="Wiki caption encoding")):
            w_prompt_raw = w_prompts_raw[idx]

            sents = corpus.get(idx, [])
            if len(sents) == 0:
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
            else:
                w_caption_raw = F.normalize(selected.mean(0), dim=-1)
                raw_trust_score = (w_prompt_raw * w_caption_raw).sum() # [-1, 1] 범위의 코사인 유사도
                trust_score = raw_trust_score.clamp(min=0.0).item()
                alpha = 1.0 - trust_score
                
                w_final = F.normalize(alpha * w_prompt_raw + (1 - alpha) * w_caption_raw, dim=-1)

            all_caption_features.append(w_final)

        # 7️⃣ 최종 classifier weight로 사용
        self.class_features = torch.stack(all_caption_features, dim=0)
        print(f"[Wiki] Done: computed features for {len(self.classnames)} classes (top-{top_k}, thresh>{sim_threshold}, dynamic_alpha, chunked).")
        
        return self.class_features
    
    def build_optimizer(self):
        cfg = self.cfg
        
        print("Turning off gradients in the model.")
        for param in self.model.parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner.")
        for param in self.tuner.parameters():
            param.requires_grad_(True)

        self.optim = torch.optim.SGD(self.tuner.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.optim.zero_grad()
        
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = torch.GradScaler("cuda") if cfg.prec_train == "amp" else None

    def _build_stage_schedule(self):
        mix_mode = getattr(self.cfg.v, "hybrid_mix_mode", "parallel")
        if mix_mode != "sequential":
            return [("joint", self.cfg.num_epochs)]

        first = getattr(self.cfg.v, "sequential_first", "lora")
        if first not in ("lora", "adaptformer"):
            raise ValueError(f"Unsupported sequential_first: {first}")
        second = "adaptformer" if first == "lora" else "lora"

        first_epochs = int(getattr(self.cfg.v, "sequential_first_epochs", 0))
        second_epochs = int(getattr(self.cfg.v, "sequential_second_epochs", 0))
        joint_epochs = int(getattr(self.cfg.v, "sequential_joint_epochs", 0))

        assigned = first_epochs + second_epochs + joint_epochs
        remain = self.cfg.num_epochs - assigned
        if remain > 0:
            joint_epochs += remain

        schedule = []
        if first_epochs > 0:
            schedule.append((first, first_epochs))
        if second_epochs > 0:
            schedule.append((second, second_epochs))
        if joint_epochs > 0:
            schedule.append(("joint", joint_epochs))
        if not schedule:
            schedule = [("joint", self.cfg.num_epochs)]
        return schedule

    def _stage_param_enabled(self, name, stage):
        if stage == "joint":
            return True
        if stage == "lora":
            return "adaptformer" not in name
        if stage == "adaptformer":
            return "lora" not in name
        raise ValueError(f"Unknown stage: {stage}")

    def _set_trainable_stage(self, stage):
        for param in self.model.parameters():
            param.requires_grad_(False)
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(self._stage_param_enabled(name, stage))

    def _rebuild_optimizer_for_stage(self, stage, num_epochs):
        cfg = self.cfg
        self._set_trainable_stage(stage)
        params = [p for p in self.tuner.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError(f"No trainable parameters selected for stage={stage}")

        self.optim = torch.optim.SGD(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum
        )
        self.optim.zero_grad()
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, max(1, num_epochs))
        if self.scaler is None and cfg.prec_train == "amp":
            self.scaler = torch.GradScaler("cuda")
        print(f"[Stage] {stage}: epochs={num_epochs}, params={sum(p.numel() for p in params)}")

    # -------------------------
    # Textual-prior regularizer
    # -------------------------
    @contextmanager
    def _disable_adapters(self, adapter_type: str):
        """Temporarily zero the output of all LoRA or AdaptFormer modules.

        adapter_type: "lora"  → disables every LoRA instance (attention Q/V + ffn_lora)
                      "adaptformer" → disables every AdaptFormer instance
        Used by class-adaptive training so head-class loss backprops only through LoRA
        and tail-class loss backprops only through AdaptFormer.
        """
        disabled = []
        for module in self.model.modules():
            cls_name = type(module).__name__
            if adapter_type == "lora" and cls_name == "LoRA":
                module._disabled = True
                disabled.append(module)
            elif adapter_type == "adaptformer" and cls_name == "AdaptFormer":
                module._disabled = True
                disabled.append(module)
        try:
            yield
        finally:
            for m in disabled:
                m._disabled = False

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

    def compute_class_features(self, prompts):
        if len(prompts) <= 1000:
            class_features = self.model.text_encoder(prompts)
        else:
            # CUDA out of memory
            prompt_splits = torch.split(prompts, 1000)
            class_features = torch.cat([self.model.text_encoder(x) for x in prompt_splits])
        return class_features
    
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
    
    # -------------------------
    # PEFT Warm-start (Stage 0)
    # -------------------------
    def _warmup_select_params(self):
        """
        Default: only warm up image_encoder tuner params (PEFT modules),
        exclude classifier/text_encoder unless user enables them.
        """
        cfg = self.cfg

        # 어떤 tuner를 warmup할지 (기본: image만)
        warm_image = bool(getattr(cfg, "PEFT_WARMUP_IMAGE", True))
        warm_text  = bool(getattr(cfg, "PEFT_WARMUP_TEXT", False))

        params = []

        # NOTE: self.tuner is nn.ParameterDict from PEFT_Model.tuner
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
        """
        Stage0: warm up PEFT modules only with (optional) KD + InfoNCE,
        without CE. After warmup, rebuild optimizer for normal training.
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
        # requires_grad 제어: 기본은 모두 off 후, warmup params만 on
        # -------------------------
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.tuner.parameters():
            p.requires_grad_(False)

        warm_params = self._warmup_select_params()
        if len(warm_params) == 0:
            print("⚠️ [Warmup] No warmup params selected. Skip.")
            self._peft_warmup_done = True
            return

        for p in warm_params:
            p.requires_grad_(True)

        # warmup optimizer (짧은 예열이라 AdamW가 안정적)
        optim_w = torch.optim.AdamW(warm_params, lr=warm_lr, weight_decay=cfg.weight_decay)

        # model forward args
        if cfg.classifier:
            model_args = {"use_classifier": True}
        else:
            # warmup에서 classifier 없으면 KD는 꺼졌고, InfoNCE만 가능
            text = self.generate_class_prompts()
            model_args = {"text": text, "is_text_feature": False}

        self.tuner.train()
        print("==============================================================")
        print(f"[Warmup] Start: lr={warm_lr}, epochs={warm_epochs}, steps={warm_steps}, "
              f"params={sum(p.numel() for p in warm_params)}")
        print("==============================================================")

        num_batches = len(self.train_loader)
        total_steps = warm_steps if warm_steps > 0 else warm_epochs * num_batches
        step = 0

        # amp scaler는 build_optimizer에서 만들어져 있을 수 있음
        scaler = self.scaler if cfg.prec_train == "amp" else None

        for epoch in range(10**9):  # steps 기반이면 epoch 수 의미 없음
            for batch_idx, (image, label) in enumerate(self.train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec_train == "amp":
                    with torch.autocast(device_type="cuda"):
                        # student logits (KD에만 필요)
                        logit = None
                        if warm_kd_loss is not None:
                            logit = self.model(image=image, **model_args)

                        loss = 0.0

                        # KD
                        if warm_kd_loss is not None:
                            with torch.no_grad():
                                text_logit = self._compute_text_prior_logits(image)
                            kd = warm_kd_loss(logit, text_logit)
                            loss = loss + text_reg_lambda * kd

                        # InfoNCE
                        if warm_nce_loss is not None:
                            feat_txt = self._compute_text_prior_feat(image)
                            nce = warm_nce_loss(feat_txt, self.text_prior_weight, label)
                            loss = loss + infonce_lambda * nce

                    scaler.scale(loss / cfg.accum_step).backward()
                    if ((step + 1) % cfg.accum_step == 0):
                        scaler.step(optim_w)
                        scaler.update()
                        optim_w.zero_grad()

                else:
                    logit = None
                    if warm_kd_loss is not None:
                        logit = self.model(image=image, **model_args)

                    loss = 0.0

                    if warm_kd_loss is not None:
                        with torch.no_grad():
                            text_logit = self._compute_text_prior_logits(image)
                        kd = warm_kd_loss(logit, text_logit)
                        loss = loss + text_reg_lambda * kd

                    if warm_nce_loss is not None:
                        feat_txt = self._compute_text_prior_feat(image)
                        nce = warm_nce_loss(feat_txt, self.text_prior_weight, label)
                        loss = loss + infonce_lambda * nce

                    (loss / cfg.accum_step).backward()
                    if ((step + 1) % cfg.accum_step == 0):
                        optim_w.step()
                        optim_w.zero_grad()

                # logging (간단히)
                if ((step + 1) % cfg.print_freq == 0) or (step == 0):
                    msg = [f"[Warmup] step {step+1}/{total_steps} loss={float(loss):.4f}"]
                    print(" ".join(msg))
                    sys.stdout.flush()

                step += 1
                if step >= total_steps:
                    break
            if step >= total_steps:
                break

        print("[Warmup] Done. Rebuild optimizer for normal training.")

        # Stage1을 위해 원래 로직 복구: tuner 전체 학습 + SGD/cosine 등
        self.build_optimizer()

        self._peft_warmup_done = True
    
    def train(self):
        cfg = self.cfg

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
            print("Generating class prompts.")
            text = self.generate_class_prompts()
            model_args = {"text": text, "is_text_feature": False}

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
        stage_schedule = self._build_stage_schedule()
        num_epochs = sum(stage_epochs for _, stage_epochs in stage_schedule)
        global_epoch_idx = 0
        global_iter_idx = 0
        total_batches = len(self.train_loader) * max(1, num_epochs)

        for stage_name, stage_epochs in stage_schedule:
            self._rebuild_optimizer_for_stage(stage_name, stage_epochs)
            for stage_epoch_idx in range(stage_epochs):
                self.tuner.train()
                end = time.time()

                num_batches = len(self.train_loader)
                for batch_idx, (image, label) in enumerate(self.train_loader):
                    image = image.to(self.device)
                    label = label.to(self.device)

                    kd_loss = None
                    nce_loss = None
                    head_ca_loss = None
                    tail_ca_loss = None

                    _use_ca = bool(getattr(cfg, "CLASS_ADAPTIVE_LOSS", False))
                    if _use_ca:
                        _ca_head_lam = float(getattr(cfg, "CLASS_ADAPTIVE_HEAD_LAMBDA", 1.0))
                        _ca_tail_lam = float(getattr(cfg, "CLASS_ADAPTIVE_TAIL_LAMBDA", 1.0))
                        _head_mask = self.many_mask[label.cpu()].to(self.device)
                        _tail_mask = self.few_mask[label.cpu()].to(self.device)

                    if cfg.prec_train == "amp":
                        with torch.autocast(device_type="cuda"):
                            logit = self.model(image=image, **model_args)
                            ce_loss = self.criterion(logit, label)

                            loss = ce_loss

                            if self.text_reg_lambda > 0:
                                with torch.no_grad():
                                    text_logit = self._compute_text_prior_logits(image)
                                kd_loss = self.text_reg_loss(logit, text_logit)
                                loss = loss + self.text_reg_lambda * kd_loss

                            if self.infonce_lambda > 0:
                                feat_txt = self._compute_text_prior_feat(image)
                                nce_loss = self.infonce_loss(feat_txt, self.text_prior_weight, label)
                                loss = loss + self.infonce_lambda * nce_loss

                            # --- Class-adaptive loss ---
                            # Head-class pass: AdaptFormer disabled → LoRA trains on head classes
                            # Tail-class pass: LoRA disabled → AdaptFormer trains on tail classes
                            if _use_ca:
                                if _head_mask.any() and _ca_head_lam > 0:
                                    with self._disable_adapters("adaptformer"):
                                        head_logit = self.model(image[_head_mask], **model_args)
                                    head_ca_loss = self.criterion(head_logit, label[_head_mask])
                                    loss = loss + _ca_head_lam * head_ca_loss

                                if _tail_mask.any() and _ca_tail_lam > 0:
                                    with self._disable_adapters("lora"):
                                        tail_logit = self.model(image[_tail_mask], **model_args)
                                    tail_ca_loss = self.criterion(tail_logit, label[_tail_mask])
                                    loss = loss + _ca_tail_lam * tail_ca_loss

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
                            kd_loss = self.text_reg_loss(logit, text_logit)
                            loss = loss + self.text_reg_lambda * kd_loss

                        if self.infonce_lambda > 0:
                            feat_txt = self._compute_text_prior_feat(image)
                            nce_loss = self.infonce_loss(feat_txt, self.text_prior_weight, label)
                            loss = loss + self.infonce_lambda * nce_loss

                        # --- Class-adaptive loss ---
                        if _use_ca:
                            if _head_mask.any() and _ca_head_lam > 0:
                                with self._disable_adapters("adaptformer"):
                                    head_logit = self.model(image[_head_mask], **model_args)
                                head_ca_loss = self.criterion(head_logit, label[_head_mask])
                                loss = loss + _ca_head_lam * head_ca_loss

                            if _tail_mask.any() and _ca_tail_lam > 0:
                                with self._disable_adapters("lora"):
                                    tail_logit = self.model(image[_tail_mask], **model_args)
                                tail_ca_loss = self.criterion(tail_logit, label[_tail_mask])
                                loss = loss + _ca_tail_lam * tail_ca_loss

                        (loss / cfg.accum_step).backward()
                        if ((batch_idx + 1) % cfg.accum_step == 0) or (batch_idx + 1 == num_batches):
                            self.optim.step()
                            self.optim.zero_grad()

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
                        nb_done = global_epoch_idx * num_batches + batch_idx + 1
                        nb_remain = max(0, total_batches - nb_done)
                        eta_seconds = batch_time.avg * nb_remain
                        eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                        info = []
                        info += [f"stage {stage_name}"]
                        info += [f"epoch [{global_epoch_idx + 1}/{num_epochs}]"]
                        info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                        info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                        info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                        info += [f"acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})"]
                        info += [f"(mean {mean_acc:.2f} many {many_acc:.2f} med {med_acc:.2f} few {few_acc:.2f})"]
                        if kd_loss is not None:
                            info += [f"kd {kd_loss.item():.4f}"]
                        if nce_loss is not None:
                            info += [f"nce {nce_loss.item():.4f}"]
                        if head_ca_loss is not None:
                            info += [f"head_ca {head_ca_loss.item():.4f}"]
                        if tail_ca_loss is not None:
                            info += [f"tail_ca {tail_ca_loss.item():.4f}"]
                        info += [f"lr {current_lr:.4e}"]
                        info += [f"eta {eta}"]
                        print(" ".join(info))
                        sys.stdout.flush()

                    tb_writer.add_scalar("train/lr", current_lr, global_iter_idx)
                    tb_writer.add_scalar("train/loss.val", loss_meter.val, global_iter_idx)
                    tb_writer.add_scalar("train/loss.avg", loss_meter.avg, global_iter_idx)
                    tb_writer.add_scalar("train/acc.val", acc_meter.val, global_iter_idx)
                    tb_writer.add_scalar("train/acc.avg", acc_meter.avg, global_iter_idx)
                    tb_writer.add_scalar("train/mean_acc", mean_acc, global_iter_idx)
                    tb_writer.add_scalar("train/many_acc", many_acc, global_iter_idx)
                    tb_writer.add_scalar("train/med_acc", med_acc, global_iter_idx)
                    tb_writer.add_scalar("train/few_acc", few_acc, global_iter_idx)
                    tb_writer.add_scalar("train/ce_loss", ce_loss.item(), global_iter_idx)
                    if kd_loss is not None:
                        tb_writer.add_scalar("train/kd_loss", kd_loss.item(), global_iter_idx)
                    if nce_loss is not None:
                        tb_writer.add_scalar("train/nce_loss", nce_loss.item(), global_iter_idx)
                    if head_ca_loss is not None:
                        tb_writer.add_scalar("train/head_ca_loss", head_ca_loss.item(), global_iter_idx)
                    if tail_ca_loss is not None:
                        tb_writer.add_scalar("train/tail_ca_loss", tail_ca_loss.item(), global_iter_idx)
                    tb_writer.add_scalar("train/stage_id", float(["lora", "adaptformer", "joint"].index(stage_name)
                                                                  if stage_name in ("lora", "adaptformer", "joint")
                                                                  else -1), global_iter_idx)

                    global_iter_idx += 1
                    end = time.time()

                self.sched.step()
                for t in self.train_loader.dataset.transform.transforms:
                    if isinstance(t, MinimalistRandomResizedCrop):
                        t.step()
                global_epoch_idx += 1

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
            model_args = {"use_classifier": True}
        else:
            print("Pre-computing class features for testing.")
            text = self.generate_class_prompts()
            with torch.no_grad():
                text = self.compute_class_features(text)
            model_args = {"text": text, "is_text_feature": True}
        
        for image, label in tqdm(self.test_loader, ascii=True, desc="Testing"):
            image = image.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if cfg.tte:  # [bsz, ncrops, C, H, W] 
                    logit = torch.stack([self.model(image=x, **model_args) for x in image.unbind(dim=1)]).mean(dim=0)
                else:
                    logit = self.model(image=image, **model_args)

            evaluator.process(logit, label)

        evaluator.evaluate(self.many_classes, self.med_classes, self.few_classes)

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
