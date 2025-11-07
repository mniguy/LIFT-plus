import os
import sys
import time
import datetime
import math
import random
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from functools import partial
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
import json, re
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
                with open(caption_path, "r", encoding="utf-8") as f:
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

            if not (cfg.zero_shot or cfg.test_only) and cfg.classifier_init is not None:
                classifier_init = cfg.classifier_init
                
                if classifier_init == "semantic":
                    print("Using semantic-aware initialization.")
                    with torch.no_grad():
                        class_features = self.compute_class_features(self.generate_class_prompts())
                    self.model.init_classifier_weight(class_features, feature_modality="text")
                

                elif classifier_init == "hybrid":
                    print("Using real-time hybrid initialization.")
                    with torch.no_grad():
                        class_features = self._compute_caption_features()
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
        for idx, cname in enumerate(tqdm(self.classnames, desc="Wiki caption encoding")):
            w_prompt_raw = w_prompts_raw[idx]

            # 2️⃣ wiki 문장 feature
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

            # 💡 --- >> 수정된 청킹(Chunking) 로직 시작 << --- 💡
            sent_feats_list = []
            
            # 128개씩 배치 처리하는 것은 유지 (메모리 관리)
            for i in range(0, len(sents), 128):
                batch_sents = sents[i:i+128] # 현재 배치(128개)의 문장들
                chunked_batch_sents = [] # 잘라낸 텍스트 조각들
                sent_indices = []        # 각 조각이 원래 몇 번째 문장 소속인지 (0~127)
                word_chunk_size = 40     # 77토큰 제한을 넘지 않기 위한 휴리스틱 (약 40단어)

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

            # 💡 --- >> 수정 6: 방법 1 (Dynamic Alpha) 적용 << --- 💡
            
            # 4️⃣ caption feature 평균 (및 Fallback 로직)
            if selected.shape[0] == 0:
                # 캡션이 없거나, top-k가 모두 threshold 미달이면 프롬프트만 사용
                alpha = 1.0
                w_final = w_prompt_raw
            else:
                # 캡션 특징 생성
                w_caption_raw = F.normalize(selected.mean(0), dim=-1)
                
                # 5️⃣ 캡션 신뢰도(유사도) 기반 동적 Alpha 계산
                # (w_prompt_raw와 w_caption_raw는 이미 정규화됨)
                trust_score = (w_prompt_raw * w_caption_raw).sum().item()
                
                # 신뢰도가 높으면 alpha가 낮아짐 (캡션 비중 증가)
                # 신뢰도가 낮으면 alpha가 높아짐 (프롬프트 비중 증가)
                alpha = 1.0 - trust_score
                
                # 6️⃣ 동적으로 계산된 alpha 비율로 혼합
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

    def train(self):
        cfg = self.cfg

        # Initialize tensorboard summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        tb_writer = SummaryWriter(log_dir=writer_dir)
        
        # Initialize average meters
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
        
        print("Start training")
        # Record the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, (image, label) in enumerate(self.train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec_train == "amp":
                    with torch.autocast(device_type="cuda"):
                        logit = self.model(image=image, **model_args)
                        loss = self.criterion(logit, label)
                    self.scaler.scale(loss / cfg.accum_step).backward()
                    if ((batch_idx + 1) % cfg.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    logit = self.model(image=image, **model_args)
                    loss = self.criterion(logit, label)
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
                med_acc = torch.mean(torch.Tensor(cls_accs)[self.med_classes])
                few_acc = torch.mean(torch.Tensor(cls_accs)[self.few_classes])
                
                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})"]
                    info += [f"(mean {mean_acc:.2f} many {many_acc:.2f} med {med_acc:.2f} few {few_acc:.2f})"]
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

                end = time.time()
            
            self.sched.step()
            for t in self.train_loader.dataset.transform.transforms:
                if isinstance(t, MinimalistRandomResizedCrop):
                    t.step()
            # torch.cuda.empty_cache()
        
        print("Finish training")
        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")
        
        # save model
        self.save_model(cfg.output_dir)

        # close writer
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
