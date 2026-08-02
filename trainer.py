import os
import sys
import time
import datetime
import math

from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from typing import List

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
from utils.samplers import ClassBalancedSampler, DownSampler
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

        # B2 intervention: "balanced" gives every class an equal share of the SAME number of
        # gradient steps (see ClassBalancedSampler). Pair with loss_type=CE -- under balanced
        # sampling the effective prior is uniform, so LA/BS/LDAM would double-correct.
        train_sampler = None
        if getattr(cfg, "train_sampler", "default") == "balanced":
            train_sampler = ClassBalancedSampler(train_dataset.labels)
            print(f"[train_sampler] class-balanced: {len(train_sampler)} samples/epoch "
                  f"(= default), equal gradient share across {self.num_classes} classes")

        self.train_loader = DataLoader(train_dataset,
            batch_size=micro_batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
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
        
    def _get_prompt_templates(self) -> List[str]:
        prompt_mode = getattr(self.cfg, "PROMPT_MODE", "default")

        if prompt_mode == "default":
            return ["a photo of a {}."]
        if prompt_mode == "bare":                 # #6 ablation: no template, just the class name itself
            return ["{}."]
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

            if not (cfg.zero_shot or cfg.test_only) and cfg.classifier_init is not None:
                classifier_init = cfg.classifier_init
                
                if classifier_init == "semantic":
                    print("Using semantic-aware initialization.")
                    with torch.no_grad():
                        class_features = self.compute_prompt_class_features()
                        if getattr(cfg, "PROMPT_CENTER", False):  # prototype centering / de-anisotropization
                            class_features = self._center_prototypes(class_features)
                            print(f"[PROMPT_CENTER] mode={getattr(cfg, 'PROMPT_CENTER_MODE', 'global')} "
                                  f"applied to prototypes.")
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

                elif classifier_init == "img_shrink":
                    # count-adaptive blend of class-mean IMAGE features and centered TEXT prototypes:
                    #   w_c = normalize(lam_c * imgmean_c + (1-lam_c) * centered_text_c),  lam_c = n_c/(n_c+kappa)
                    # head (many imgs) -> trust image mean; tail (few) -> fall back to centered text.
                    kappa = float(getattr(cfg, "IMG_SHRINK_KAPPA", 20.0))
                    print(f"Using image-mean + shrink-to-centered-text init (kappa={kappa}).")
                    with torch.no_grad():
                        # centered text prototypes, projected into the image/classifier space
                        text = self._center_prototypes(self.compute_prompt_class_features())
                        text_img = F.normalize(text @ self.model.text_proj.data, dim=-1)
                        if hasattr(self.model, "image_proj"):
                            text_img = F.normalize(text_img @ self.model.image_proj.data.t(), dim=-1)
                        # class-mean image features
                        train_features, train_labels = self.compute_train_features()
                        sorted_index = train_labels.argsort()
                        train_features = train_features[sorted_index]; train_labels = train_labels[sorted_index]
                        _, label_counts = torch.unique(train_labels, return_counts=True)
                        img_means = F.normalize(torch.stack(
                            [x.mean(dim=0) for x in torch.split(train_features, label_counts.tolist())]), dim=-1)
                        cn = torch.as_tensor(self.cls_num_list, dtype=torch.float32, device=img_means.device)
                        lam = (cn / (cn + kappa)).unsqueeze(1)                       # [C,1]
                        blend = lam * img_means + (1 - lam) * text_img.to(img_means.device)
                    self.model.init_classifier_weight(blend, feature_modality="image")
                    print(f"[img_shrink] lambda in [{lam.min().item():.2f}, {lam.max().item():.2f}] "
                          f"(head->imagemean, tail->centered-text)")

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

    def _load_taxonomy(self):
        """classname -> its categories.json record (kingdom/phylum/class/order/family/genus).

        iNat only; returns None for ImageNet/Places (no categories.json), which makes
        PROMPT_CENTER_MODE=cascade an explicit error there rather than a silent no-op.
        """
        cats_path = os.path.join("datasets", self.cfg.dataset, "categories.json")
        if not os.path.exists(cats_path):
            return None
        import json
        try:
            cats = json.load(open(cats_path))
            return {c["name"]: c for c in cats if "name" in c}
        except (ValueError, TypeError, KeyError):
            return None

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
        elif mode == "kappa":                    # smooth COUNT-based strength (vs tail's rank-based ramp):
            # rarity_c = kappa/(n_c+kappa) -- reuses the img_shrink shrinkage form, same read: rare (n->0)
            # -> rarity->1 (full center), common (n->inf) -> rarity->0. kappa sets the half-strength count.
            kappa = float(getattr(self.cfg, "PROMPT_CENTER_KAPPA", 20.0))
            cn = torch.as_tensor(self.cls_num_list, dtype=torch.float32, device=X.device)
            rarity = (kappa / (cn + kappa)).unsqueeze(1)               # [C,1] rare->1, common->0
            out = X - rarity * X.mean(0)
        elif mode == "logcount":                 # parameter-free: LINEAR ramp in log-count space (vs tail's
            # linear-in-RANK, which ignores count gaps, and kappa's hyperbolic-in-count, whose shape is tied
            # to kappa). rarity_c = (log(n_max)-log(n_c)) / (log(n_max)-log(n_min)), clipped to [0,1].
            cn = torch.as_tensor(self.cls_num_list, dtype=torch.float32, device=X.device)
            log_n = cn.clamp_min(1.0).log()
            lo, hi = log_n.min(), log_n.max()
            rarity = ((hi - log_n) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0).unsqueeze(1)
            out = X - rarity * X.mean(0)
        elif mode == "genus":                    # taxonomy-aware LOCAL group (iNat): subtract the per-genus
            # mean instead of one global mu. Genus = first token of classnames (binomial "Genus species",
            # true for iNat's default name-based classnames; degrades to global on non-binomial datasets
            # since every class becomes its own singleton "genus"). Genera smaller than the min size fall
            # back to global mu -- most iNat genera are singletons (68% have exactly 1 species; subtracting
            # a genus's own single-member mean would zero that class's vector out entirely).
            min_size = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            global_mu = X.mean(0)
            genus_of = [name.split()[0] for name in self.classnames]
            groups_idx = {}
            for i, g in enumerate(genus_of):
                groups_idx.setdefault(g, []).append(i)
            local_mu = global_mu.unsqueeze(0).repeat(X.shape[0], 1)
            n_fallback = 0
            for g, idxs in groups_idx.items():
                if len(idxs) >= min_size:
                    idxs_t = torch.as_tensor(idxs, device=X.device)
                    local_mu[idxs_t] = X[idxs_t].mean(0)
                else:
                    n_fallback += len(idxs)
            print(f"[PROMPT_CENTER genus] {n_fallback}/{len(genus_of)} classes fell back to global mu "
                  f"(genus size < {min_size}); {len(groups_idx)} distinct genera")
            out = X - local_mu
        elif mode == "cascade":                  # HIERARCHICAL taxonomy fallback (iNat): try the deepest
            # level first, and only the classes still unassigned drop to the next level up, so nothing has
            # to fall all the way to global. Fixes 'genus' mode's coverage hole (only 28% of iNat species
            # sit in a genus >= 5, the other 72% jumped straight to global and kept their genus blob).
            # PROMPT_CENTER_CASCADE_MEAN picks what a fallback level's mean is taken over:
            #   "residual" (default): only the STILL-UNASSIGNED members -- the shared component left
            #      among the classes that actually still need one, and the group must have >= min_size
            #      of THOSE to qualify.
            #   "full": every member of the group, deeper-assigned ones included (a 4-species genus in
            #      an 11-species family subtracts the whole family's mean, not the mean of the 4 left
            #      over). Group size is then the full group size, so more classes get a taxonomic mean
            #      and fewer reach global.
            levels = [s.strip() for s in
                      str(getattr(self.cfg, "PROMPT_CENTER_CASCADE", "genus,family,order")).split(",") if s.strip()]
            min_size = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            mean_mode = str(getattr(self.cfg, "PROMPT_CENTER_CASCADE_MEAN", "residual"))
            if mean_mode not in ("residual", "full"):
                raise ValueError(f"unknown PROMPT_CENTER_CASCADE_MEAN: {mean_mode}")
            taxo = self._load_taxonomy()
            if taxo is None:
                raise ValueError("PROMPT_CENTER_MODE=cascade needs a dataset with categories.json (iNat only)")
            local_mu = X.mean(0).unsqueeze(0).repeat(X.shape[0], 1)
            assigned = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
            used = []
            for lv in levels:
                groups_idx = {}
                for i, name in enumerate(self.classnames):
                    if assigned[i] and mean_mode == "residual":
                        continue
                    key = taxo.get(name, {}).get(lv)
                    if key is not None:
                        groups_idx.setdefault(key, []).append(i)
                n_lv = 0
                for key, idxs in groups_idx.items():
                    if len(idxs) < min_size:
                        continue
                    idxs_t = torch.as_tensor(idxs, device=X.device)
                    target = idxs_t[~assigned[idxs_t]]      # only classes without a mean yet get one
                    if target.numel() == 0:
                        continue
                    local_mu[target] = X[idxs_t].mean(0)    # mean over the group as scoped above
                    assigned[target] = True
                    n_lv += int(target.numel())
                used.append(f"{lv}={n_lv}")
            used.append(f"global={int((~assigned).sum())}")
            print(f"[PROMPT_CENTER cascade] levels={levels} min_size={min_size} mean={mean_mode} -> "
                  + " ".join(used))
            out = X - local_mu
        elif mode == "knn":                      # per-class LOCAL group via k-nearest classes (taxonomy-free
            # generalization of 'genus' -- works on any dataset). Subtracts the mean of each class's k
            # nearest OTHER classes (by prototype cosine similarity) instead of one fixed global mu.
            k = int(getattr(self.cfg, "PROMPT_CENTER_KNN_K", 20))
            Xn = F.normalize(X, dim=-1)
            sim = Xn @ Xn.t()
            sim.fill_diagonal_(-2.0)             # exclude self from its own neighbor list
            topk_idx = sim.topk(min(k, X.shape[0] - 1), dim=1).indices   # [C, k]
            local_mu = X[topk_idx].mean(dim=1)   # [C, D] mean of the k nearest OTHER classes
            out = X - local_mu
        elif mode == "cluster":                  # semantic k-means groups (taxonomy-free, unsupervised):
            # partition the prototypes into PROMPT_CENTER_CLUSTER_K clusters and subtract each class's
            # own cluster mean. Hard-partition counterpart of 'knn' (which uses a per-class, overlapping
            # neighborhood) and the taxonomy-free stand-in for 'cascade' -- on iNat the cluster/family
            # agreement can be measured, on IN/PL it is the only local option. Clusters smaller than
            # PROMPT_CENTER_GENUS_MIN fall back to global mu (same guard as genus/cascade).
            from sklearn.cluster import KMeans
            k = int(getattr(self.cfg, "PROMPT_CENTER_CLUSTER_K", 100))
            min_size = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            target = int(getattr(self.cfg, "PROMPT_CENTER_CLUSTER_SIZE", 0))
            if target > 0:                   # dataset-relative: fix the GRANULARITY (avg classes per
                # cluster) instead of the cluster count, so one setting means the same thing on 365,
                # 1000 and 8142 classes. k=100 is 3 classes/cluster on Places and 57 on iNat.
                k = max(1, round(X.shape[0] / target))
            k = min(k, X.shape[0])
            labels = KMeans(n_clusters=k, n_init=10, random_state=int(getattr(self.cfg, "seed", 0))
                            ).fit_predict(F.normalize(X, dim=-1).cpu().numpy())
            labels = torch.as_tensor(labels, device=X.device)
            local_mu = X.mean(0).unsqueeze(0).repeat(X.shape[0], 1)
            n_fallback = 0
            for c in range(k):
                idxs = (labels == c).nonzero(as_tuple=True)[0]
                if idxs.numel() >= min_size:
                    local_mu[idxs] = X[idxs].mean(0)
                else:
                    n_fallback += int(idxs.numel())
            print(f"[PROMPT_CENTER cluster] k={k} (target size={target or '-'}) min_size={min_size} "
                  f"-> {n_fallback}/{X.shape[0]} classes fell back to global mu")
            out = X - local_mu
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

    def train(self):
        cfg = self.cfg

        # Save initial (pre-training) checkpoint for visualization
        init_dir = os.path.join(cfg.output_dir, "ckpts", "init")
        os.makedirs(init_dir, exist_ok=True)
        self.save_model(init_dir)

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

        print("Start training")
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
                        ce_loss = self.criterion(logit, label)

                        loss = ce_loss

                    self.scaler.scale(loss / cfg.accum_step).backward()
                    if ((batch_idx + 1) % cfg.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()

                else:
                    logit = self.model(image=image, **model_args)
                    ce_loss = self.criterion(logit, label)

                    loss = ce_loss

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
