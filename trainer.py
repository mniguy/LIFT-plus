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

    def _level_mean(self, X, lv, taxo):
        """[C, D] per-class mean of its group at taxonomy level `lv` (self included).

        Used by the two modes that combine level means linearly (blend, sum_all). The older
        taxonomy modes build this inline; they are deliberately left untouched.
        """
        mu = torch.zeros_like(X)
        groups = {}
        for i, name in enumerate(self.classnames):
            key = taxo.get(name, {}).get(lv)
            if key is None:
                raise ValueError(f"taxonomy level '{lv}' missing for class '{name}'")
            groups.setdefault(key, []).append(i)
        for _, idxs in groups.items():
            idxs_t = torch.as_tensor(idxs, device=X.device)
            mu[idxs_t] = X[idxs_t].mean(0)
        return mu

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
        elif mode == "genus_lex":                # D: surgical LEXICAL fix (vs 'genus's raw group mean).
            # 'genus' subtracts the mean of genus-mates' FULL embeddings, which conflates two things:
            # (a) the repeated genus WORD itself, and (b) whatever genuine within-genus content those
            # particular species happen to share beyond the word. This isolates (a) specifically: encode
            # each class a second time with the genus word stripped (species epithet alone), take the
            # per-class difference embed(full) - embed(epithet) as that class's estimate of "what the
            # genus word alone contributes", and subtract the GENUS-AVERAGE of that difference vector --
            # not the genus-average of the full embedding. Classes whose genus is too small fall back to
            # plain global mu, same guard as 'genus'.
            min_size = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            global_mu = X.mean(0)
            genus_of = [name.split()[0] for name in self.classnames]
            epithets = [" ".join(name.split()[1:]) if len(name.split()) > 1 else name
                        for name in self.classnames]
            with torch.no_grad():
                epi_prompts = clip.tokenize(
                    [self.template.format(e.replace("_", " ")) for e in epithets]).to(X.device)
                # NO F.normalize here: X arrives as raw CLIP text features (norm ~8.6 on B/16, since
                # compute_prompt_class_features' single-template path does not normalize). Unit-norming
                # only the epithet side made the subtraction ~20x too small -- measured cos(diff, X)
                # = 0.9968, i.e. "diff" was just X and the lexical isolation was a no-op.
                X_epi = self.compute_class_features(epi_prompts).float()
            diff = X - X_epi                      # [C, D] per-class estimate of the genus word's own contribution
            groups_idx = {}
            for i, g in enumerate(genus_of):
                groups_idx.setdefault(g, []).append(i)
            local_mu = global_mu.unsqueeze(0).repeat(X.shape[0], 1)
            n_fallback = 0
            for g, idxs in groups_idx.items():
                if len(idxs) >= min_size:
                    idxs_t = torch.as_tensor(idxs, device=X.device)
                    local_mu[idxs_t] = diff[idxs_t].mean(0)     # subtract the LEXICAL diff, not the raw group mean
                else:
                    n_fallback += len(idxs)
            print(f"[PROMPT_CENTER genus_lex] {n_fallback}/{len(genus_of)} classes fell back to global mu "
                  f"(genus size < {min_size}); {len(groups_idx)} distinct genera; "
                  f"mean|diff|={diff.norm(dim=-1).mean().item():.3f}")
            out = X - local_mu
        elif mode == "diff_init":                # A': the lexical diff IS the initialization.
            # Every other lexical mode uses diff only to build something to SUBTRACT from X. This one
            # discards X entirely and initializes from diff = embed(full binomial) - embed(epithet
            # alone), then globally centers it. Rationale from an offline measurement on all 8142 real
            # iNat prototypes: diff is NOT the genus-shared blob one would expect (the text encoder is
            # non-linear, so what the genus word contributes depends on the species it is paired with).
            # Measured within-genus cosine: raw 0.944, global-centered 0.834, but diff-then-centered
            # 0.624 -- the lowest of anything tried -- and top5-conf (mean cosine to each class's 5
            # nearest OTHER classes, iNat's actual confusion bottleneck) 0.557 vs global's 0.668.
            # RISK, stated up front: this replaces the semantic target geometry ("what this species
            # is") with a lexical one ("what the genus word contributes"), so the image encoder has a
            # much larger remapping to learn. cos to the global-centered init is only 0.560, far from
            # the 0.72-0.75 band where every arm that has actually won on this dataset sits. Note also
            # that top5-conf has NOT predicted accuracy here before (the family level had the best
            # top5-conf in the level ladder yet only middling accuracy), so the geometry above is a
            # reason to test this, not a reason to expect it to win.
            epithets = [" ".join(name.split()[1:]) if len(name.split()) > 1 else name
                        for name in self.classnames]
            with torch.no_grad():
                epi_prompts = clip.tokenize(
                    [self.template.format(e.replace("_", " ")) for e in epithets]).to(X.device)
                X_epi = self.compute_class_features(epi_prompts).float()   # raw scale, see genus_lex note
            diff = X - X_epi
            print(f"[PROMPT_CENTER diff_init] mean|X|={X.norm(dim=-1).mean().item():.3f} "
                  f"mean|X_epi|={X_epi.norm(dim=-1).mean().item():.3f} "
                  f"mean|diff|={diff.norm(dim=-1).mean().item():.3f} "
                  f"cos(diff,X)={(F.normalize(diff, dim=-1) * F.normalize(X, dim=-1)).sum(-1).mean().item():.4f} "
                  f"(expect ~0.38 with mean|X| ~23; ~1.0 would mean the epithet subtraction was a no-op)")
            out = diff - diff.mean(0)             # global centering of the diff == variant A'
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
            # PROMPT_CENTER_CASCADE_GLOBAL_FIRST: subtract the global centroid before the cascade runs.
            # This is a PROVABLE NO-OP, kept only as a null control. Mean subtraction is linear, so the
            # global term cancels out of every group mean computed afterwards:
            #   (X - mu_g) - [mu_group(X) - mu_g] = X - mu_group(X)
            # and the classes that reach the global fallback see a residual whose global mean is already
            # zero. Verified offline against plain cascade: per-class cosine 1.00000000 (min 0.99999964),
            # max elementwise difference 2e-7, i.e. float32 rounding. Any accuracy difference this
            # produces is therefore a measurement of run-to-run variation, not of the operation.
            if bool(getattr(self.cfg, "PROMPT_CENTER_CASCADE_GLOBAL_FIRST", False)):
                X = X - X.mean(0)
                print("[PROMPT_CENTER cascade] global centroid removed first "
                      "(provable no-op; null control for run-to-run variation)")
            # PROMPT_CENTER_CASCADE_NOFALL: what a class that qualifies at NO level receives.
            # Default (False) is the global mean -- the "fallback" in this mode has never meant
            # "left alone", it means "centered with the global centroid instead of a local one".
            # With True the fallback vector is ZERO, so such a class keeps its RAW prototype and is
            # genuinely not centered at all. Safe (raw O is far from the origin, unlike the zero rows
            # that broke mode=level), but it makes the init inhomogeneous: those classes end up where
            # mode=shrink leaves its singletons.
            nofall = bool(getattr(self.cfg, "PROMPT_CENTER_CASCADE_NOFALL", False))
            local_mu = (torch.zeros_like(X) if nofall
                        else X.mean(0).unsqueeze(0).repeat(X.shape[0], 1))
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
            used.append(f"{'UNCENTERED' if nofall else 'global'}={int((~assigned).sum())}")
            print(f"[PROMPT_CENTER cascade] levels={levels} min_size={min_size} mean={mean_mode} nofall={nofall} -> "
                  + " ".join(used))
            out = X - local_mu
            # PROMPT_CENTER_CASCADE_GLOBAL_LAST: remove the residual's global centroid AFTER cascading.
            # Also a near no-op, but for a different reason than GLOBAL_FIRST. Every class assigned to a
            # real group already has that group's mean zeroed, so the only thing leaving a nonzero
            # overall mean is the global-fallback set, which subtracts the mean over ALL classes rather
            # than over itself. Measured on the real prototypes: the leftover centroid has norm 0.0031
            # against an original 0.8278 (0.8% of the mean row norm), and the resulting init is
            # per-class cosine 0.999937 to plain cascade (min 0.999036). Paired with GLOBAL_FIRST
            # (cosine 1.0000000) this gives two null controls at slightly different distances from
            # exact identity.
            if bool(getattr(self.cfg, "PROMPT_CENTER_CASCADE_GLOBAL_LAST", False)):
                res_mu = out.mean(0)
                print(f"[PROMPT_CENTER cascade] global centroid removed last: |mu_residual|="
                      f"{res_mu.norm().item():.4f} vs |mu_original|={X.mean(0).norm().item():.4f} "
                      f"(near no-op; null control)")
                out = out - res_mu
        elif mode == "cascade_lex":               # genus_lex's surgical diff-vector subtraction,
            # plugged into cascade's multi-level fallback instead of standing alone. Only the 'genus'
            # level has a literal shared TOKEN to isolate this way (binomial "Genus species" repeats
            # the genus word; a family name like "Fagaceae" never appears inside a classname at all,
            # so there is no analogous epithet-style split for family/order) -- those levels fall back
            # to ordinary full-embedding-mean subtraction, identical to plain 'cascade'. Same
            # levels/min_size/PROMPT_CENTER_CASCADE_MEAN options as 'cascade'.
            levels = [s.strip() for s in
                      str(getattr(self.cfg, "PROMPT_CENTER_CASCADE", "genus,family,order")).split(",") if s.strip()]
            min_size = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            mean_mode = str(getattr(self.cfg, "PROMPT_CENTER_CASCADE_MEAN", "residual"))
            if mean_mode not in ("residual", "full"):
                raise ValueError(f"unknown PROMPT_CENTER_CASCADE_MEAN: {mean_mode}")
            taxo = self._load_taxonomy()
            if taxo is None:
                raise ValueError("PROMPT_CENTER_MODE=cascade_lex needs a dataset with categories.json (iNat only)")
            epithets = [" ".join(name.split()[1:]) if len(name.split()) > 1 else name
                        for name in self.classnames]
            with torch.no_grad():
                epi_prompts = clip.tokenize(
                    [self.template.format(e.replace("_", " ")) for e in epithets]).to(X.device)
                X_epi = self.compute_class_features(epi_prompts).float()   # raw scale, see genus_lex note
            diff = X - X_epi                      # reused ONLY for the 'genus' level below
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
                source = diff if lv == "genus" else X     # only genus gets the lexical-diff treatment
                for key, idxs in groups_idx.items():
                    if len(idxs) < min_size:
                        continue
                    idxs_t = torch.as_tensor(idxs, device=X.device)
                    target = idxs_t[~assigned[idxs_t]]
                    if target.numel() == 0:
                        continue
                    local_mu[target] = source[idxs_t].mean(0)
                    assigned[target] = True
                    n_lv += int(target.numel())
                used.append(f"{lv}{'(lex)' if lv == 'genus' else ''}={n_lv}")
            used.append(f"global={int((~assigned).sum())}")
            print(f"[PROMPT_CENTER cascade_lex] levels={levels} min_size={min_size} mean={mean_mode} -> "
                  + " ".join(used))
            out = X - local_mu
        elif mode == "nested":                    # REPEATED centering down (or up) the taxonomy, instead
            # of cascade's "each class is centered at exactly ONE level". Every class is centered at
            # EVERY level it has a big-enough group for, so the subtractions stack.
            # PROMPT_CENTER_NESTED_LEVELS is applied IN THE ORDER GIVEN, and that order IS the direction:
            #   "order,family,genus" = top-down  (coarse -> fine)
            #   "genus,family,order" = bottom-up (fine -> coarse)
            # The pseudo-level "global" (one centroid over ALL classes, no taxonomy lookup, no min_size
            # gate) may be used anywhere in the chain. Its purpose is to DECONFOUND the direction
            # comparison: measured offline that whichever level runs FIRST also absorbs the global
            # component (|mu| 0.84-0.98), so a bare top-down-vs-bottom-up contrast conflates "which
            # direction" with "which group size estimated the global component". Prepending "global"
            # equalizes that -- cos(topdown, bottomup) rises 0.7382 -> 0.9246 once it is prepended.
            # PROMPT_CENTER_NESTED_MEAN decides what each level's mean is computed on:
            #   "recompute" (default): the CURRENT residual, so each level removes only what the levels
            #      before it did not already explain (an ANOVA-style hierarchical decomposition, and the
            #      only variant where the two directions are genuinely different operations).
            #   "static": every level's mean comes from the ORIGINAL raw prototypes and they are summed,
            #      so the same shared component is subtracted once per level. Deliberately the
            #      over-subtraction control: measured offline that this pushes many rows PAST the origin
            #      (pre-normalization norm mean 1.13 / max 2.02 vs raw 1.0), i.e. past "removed" into
            #      "negated". Direction is irrelevant here (addition commutes) -- expected to be bad.
            levels = [s.strip() for s in
                      str(getattr(self.cfg, "PROMPT_CENTER_NESTED_LEVELS", "order,family,genus")).split(",") if s.strip()]
            min_size = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            mean_mode = str(getattr(self.cfg, "PROMPT_CENTER_NESTED_MEAN", "recompute"))
            if mean_mode not in ("recompute", "static"):
                raise ValueError(f"unknown PROMPT_CENTER_NESTED_MEAN: {mean_mode}")
            # Row-renormalize after EVERY level instead of only at the end. This is the one cheap way
            # to break the telescoping identity: with a single trailing normalize, coarse-to-fine mean
            # subtraction sums to "subtract the finest group mean" exactly (verified: per-class
            # cos(topdown3, cascade) = 1.0000 on all fully-covered classes), so the intermediate levels
            # contribute nothing. Rescaling each row between levels is nonlinear and per-row, so the
            # levels stop collapsing. Measured offline on the real prototypes, it also repairs
            # bottom-up's redundant subtraction: within-genus cosine 0.2563 -> 0.0749.
            renorm = bool(getattr(self.cfg, "PROMPT_CENTER_NESTED_RENORM", False))
            if min_size < 2:
                raise ValueError("PROMPT_CENTER_GENUS_MIN must be >= 2 for mode=nested: a singleton "
                                 "group's mean is the class itself, so subtracting it zeroes the row")
            taxo = self._load_taxonomy()
            if taxo is None:
                raise ValueError("PROMPT_CENTER_MODE=nested needs a dataset with categories.json (iNat only)")
            X_out = X.clone()
            used = []
            for lv in levels:
                src = X_out if mean_mode == "recompute" else X
                shift = torch.zeros_like(X_out)
                if lv == "global":                # pseudo-level: every class, one centroid, no gate
                    shift[:] = src.mean(0)
                    n_hit = X.shape[0]
                    mag = shift.norm(dim=-1)
                    used.append(f"global(n={n_hit},|mu|={mag.mean().item():.3f})")
                    X_out = X_out - shift
                    if renorm:
                        X_out = F.normalize(X_out, dim=-1)
                    continue
                groups_idx = {}
                for i, name in enumerate(self.classnames):
                    key = taxo.get(name, {}).get(lv)
                    if key is not None:
                        groups_idx.setdefault(key, []).append(i)
                n_hit = 0
                for key, idxs in groups_idx.items():
                    if len(idxs) < min_size:
                        continue                  # too small (incl. every singleton): skip this level
                    idxs_t = torch.as_tensor(idxs, device=X.device)
                    shift[idxs_t] = src[idxs_t].mean(0)
                    n_hit += len(idxs)
                mag = shift.norm(dim=-1)
                mag_hit = mag[mag > 0].mean().item() if bool((mag > 0).any()) else 0.0
                used.append(f"{lv}(n={n_hit},|mu|={mag_hit:.3f})")
                X_out = X_out - shift
                if renorm:
                    X_out = F.normalize(X_out, dim=-1)
            print(f"[PROMPT_CENTER nested] levels={levels} min_size={min_size} mean={mean_mode} "
                  f"renorm={renorm} -> "
                  + " ".join(used)
                  + f" | pre-norm row norm mean={X_out.norm(dim=-1).mean().item():.3f} "
                    f"min={X_out.norm(dim=-1).min().item():.3f} max={X_out.norm(dim=-1).max().item():.3f}")
            out = X_out
        elif mode in ("level", "level_keep"):     # SINGLE taxonomy level, NO fallback and NO min_size gate.
            # Every taxonomy mode above (genus/cascade/nested) guards small groups, because a group of
            # size 1 has mean == the class itself and O - mu annihilates the row. These two modes remove
            # that guard on purpose, to measure what the guard was actually buying:
            #   "level":      out = O - mu(group)          singletons land exactly on 0 (see the warning
            #                                              below; 3000/8142 iNat classes at level=genus).
            #   "level_keep": out = 2*O - mu(group)        singletons degrade to the RAW prototype O
            #                                              instead of 0, applied uniformly to all classes.
            # PROMPT_CENTER_LEVEL picks the level; "global" is the whole-dataset centroid (no taxonomy
            # lookup, no groups), which for mode=level is identical to PROMPT_CENTER_MODE=global and so is
            # only worth running under level_keep.
            # NOTE on level_keep's scale: the rows are renormalized at the end, so 2*O - mu points the same
            # direction as O - 0.5*mu. level_keep is therefore exactly the half-strength point of the
            # shrinkage family O - alpha*mu, with level being alpha = 1.
            lv = str(getattr(self.cfg, "PROMPT_CENTER_LEVEL", "genus"))
            keep = (mode == "level_keep")
            if lv == "global":
                local_mu = X.mean(0).unsqueeze(0).repeat(X.shape[0], 1)
                n_groups, n_single = 1, 0
            else:
                taxo = self._load_taxonomy()
                if taxo is None:
                    raise ValueError(f"PROMPT_CENTER_MODE={mode} with PROMPT_CENTER_LEVEL={lv} needs a "
                                     "dataset with categories.json (iNat only); use LEVEL=global otherwise")
                keys = [taxo.get(name, {}).get(lv) for name in self.classnames]
                missing = [n for n, k in zip(self.classnames, keys) if k is None]
                if missing:                       # no fallback exists in these modes -- fail loud
                    raise ValueError(f"PROMPT_CENTER_LEVEL={lv} is missing for {len(missing)} classes "
                                     f"(e.g. {missing[:3]}); these modes have no fallback by design")
                groups_idx = {}
                for i, k in enumerate(keys):
                    groups_idx.setdefault(k, []).append(i)
                local_mu = torch.zeros_like(X)
                for k, idxs in groups_idx.items():
                    idxs_t = torch.as_tensor(idxs, device=X.device)
                    local_mu[idxs_t] = X[idxs_t].mean(0)
                n_groups = len(groups_idx)
                n_single = sum(len(v) for v in groups_idx.values() if len(v) == 1)
            out = (2.0 * X - local_mu) if keep else (X - local_mu)
            norms = out.norm(dim=-1)
            n_zero = int((norms < 1e-6).sum())
            print(f"[PROMPT_CENTER {mode}] level={lv} groups={n_groups} "
                  f"classes_in_singleton_group={n_single} -> pre-norm row norm "
                  f"mean={norms.mean().item():.3f} min={norms.min().item():.3f} "
                  f"max={norms.max().item():.3f}; {n_zero}/{X.shape[0]} rows are ZERO")
            if n_zero:                            # expected for mode=level, impossible for level_keep
                print(f"[PROMPT_CENTER {mode}] WARNING: {n_zero} classifier rows initialize to the zero "
                      "vector (F.normalize leaves them at 0), so those classes start with a dead logit.")
        elif mode == "taxo_kernel":              # SOFT taxonomic neighbourhood: no branch, no min_size,
            # no fallback chain. Every class subtracts a kernel-weighted mean of its RELATIVES, where
            # the weight decays geometrically with taxonomic distance:
            #     mu_i = sum_{j != i} gamma^d(i,j) O_j  /  sum_{j != i} gamma^d(i,j)
            # d(i,j) = the level at which i and j first share an ancestor: same genus 1, family 2,
            # order 3, class 4, phylum 5, kingdom 6, unrelated 7.
            #
            # WHY THIS EXISTS: every other taxonomy mode makes a BINARY membership test, which creates
            # the degenerate state "my group has 0 other members" and then patches it after the fact
            # (min_size guard / cascade's fallback chain / level_keep's +O tax on all 8142 classes).
            # Here the d=1 term simply DROPS OUT of both numerator and denominator when a class has no
            # genus-mates, and the nearest non-empty level takes over automatically. Measured on iNat:
            # Quercus agrifolia (28-species genus) puts 99.3% of its weight on its 27 genus-mates,
            # while Abaeis nicippe (singleton genus) automatically splits 55.9% family / 43.3% order.
            # Excluding self also makes a zero row structurally impossible: mu_i is a mean of OTHER
            # classes. Measured min pre-norm row norm 1.96-10.5 against a raw row norm of ~23.
            #
            # gamma is the only knob and it subsumes the existing modes: gamma -> 1 weights every class
            # equally (== mode=global; measured cos-to-global 0.9997 at gamma=0.9), gamma -> 0 keeps
            # only the nearest non-empty relatives (== cascade with min_size=2, but gateless). Offline
            # on the real prototypes: gamma 0/0.01/0.02/0.03/0.05/0.1 -> cos-to-global
            # 0.544/0.659/0.709/0.746/0.802/0.887, top5conf 0.417/0.423/0.431/0.438/0.451/0.492.
            # gamma=0.03 lands in the 0.72-0.75 band where every arm that has won on iNat sits, with a
            # top5conf far below anything else measured here (global 0.654, cascade ~0.60).
            #
            # NOTE the leave-one-out identity that motivates excluding self: for any group of size
            # k >= 2, O_i - mu^(-i) = k/(k-1) * (O_i - mu), a POSITIVE scalar multiple, so after row
            # normalization self-exclusion changes nothing (verified: per-class cos 1.0000000000 on all
            # 5142 non-singleton iNat classes). k = 1 is the only genuine singularity, which is exactly
            # the state this kernel formulation never enters.
            gamma = float(getattr(self.cfg, "PROMPT_CENTER_GAMMA", 0.03))
            taxo = self._load_taxonomy()
            if taxo is None:
                raise ValueError("PROMPT_CENTER_MODE=taxo_kernel needs a dataset with categories.json")
            LEVELS = ["genus", "family", "order", "class", "phylum", "kingdom"]
            C = X.shape[0]
            Xd = X.double()                       # gamma**7 ~ 2e-11; accumulate in fp64 for headroom
            # S_d = the set of classes within distance d. The taxonomy is NESTED (genus in family in
            # order ...), so S_d IS the level-d group and the classes at distance EXACTLY d are
            # S_d \ S_{d-1}. That telescoping is what makes this 7 group-sum passes instead of a
            # C x C distance matrix.
            S_sum = [Xd.clone()]                                            # S_0 = {i}
            S_cnt = [torch.ones(C, dtype=torch.float64, device=X.device)]
            for lv in LEVELS:
                groups = {}
                for i, name in enumerate(self.classnames):
                    key = taxo.get(name, {}).get(lv)
                    if key is None:
                        raise ValueError(f"taxo_kernel: level '{lv}' missing for class '{name}'")
                    groups.setdefault(key, []).append(i)
                s = torch.zeros_like(Xd)
                c = torch.zeros(C, dtype=torch.float64, device=X.device)
                for _, idxs in groups.items():
                    it = torch.as_tensor(idxs, device=X.device)
                    s[it] = Xd[it].sum(0)
                    c[it] = float(len(idxs))
                S_sum.append(s)
                S_cnt.append(c)
            S_sum.append(Xd.sum(0).expand_as(Xd).clone())                   # S_7 = every class
            S_cnt.append(torch.full((C,), float(C), dtype=torch.float64, device=X.device))

            near = torch.zeros(C, dtype=torch.long, device=X.device)        # nearest non-empty distance
            for dd in range(7, 0, -1):
                near[(S_cnt[dd] - S_cnt[dd - 1]) > 0] = dd
            if gamma <= 0.0:                      # limit: mean of the nearest non-empty relatives only
                mu = torch.zeros_like(Xd)
                for dd in range(1, 8):
                    hit = near == dd
                    if hit.any():
                        cnt = (S_cnt[dd] - S_cnt[dd - 1])[hit]
                        mu[hit] = (S_sum[dd] - S_sum[dd - 1])[hit] / cnt.unsqueeze(1)
            else:
                num = torch.zeros_like(Xd)
                den = torch.zeros(C, dtype=torch.float64, device=X.device)
                for dd in range(1, 8):
                    w = gamma ** dd
                    num += w * (S_sum[dd] - S_sum[dd - 1])
                    den += w * (S_cnt[dd] - S_cnt[dd - 1])
                mu = num / den.unsqueeze(1)
            out = (Xd - mu).to(X.dtype)
            nrm = out.norm(dim=-1)
            census = " ".join(f"{lv}={int((near == d).sum())}"
                              for d, lv in enumerate(LEVELS + ["unrelated"], start=1)
                              if int((near == d).sum()) > 0)
            print(f"[PROMPT_CENTER taxo_kernel] gamma={gamma if gamma > 0 else 0.0}"
                  f"{' (nearest-relative limit)' if gamma <= 0 else ''} | nearest non-empty level: "
                  f"{census} | pre-norm row norm mean={nrm.mean().item():.3f} "
                  f"min={nrm.min().item():.3f} max={nrm.max().item():.3f} | "
                  f"{int((nrm < 1e-6).sum())}/{C} rows are ZERO (must be 0 by construction)")
        elif mode == "blend":                    # LINEAR BLEND of the global mean and one level mean:
            #     out = O - (1-s) * mu_global - s * mu_LEVEL
            # Derived from the hierarchical (ANOVA) decomposition of a prototype,
            #     O = mu_global + sum_k e_k + e_species,   e_k = mu_k - mu_{k-1}
            # by shrinking EVERY level effect by the same factor s. The telescoping sum then folds up
            # into the two-term form above, so what looks like six shrinkage terms is really one knob.
            # s = 0 is exactly mode=global; s -> 1 approaches mode=level at the same LEVEL.
            #
            # WHY THIS IS SINGLETON-SAFE WITHOUT A BRANCH: if the class is alone in its group then
            # mu_LEVEL = O, and the expression collapses to (1-s) * (O - mu_global) -- a positive
            # multiple of the globally centered vector, i.e. that class simply receives GLOBAL
            # centering. No guard, no fallback chain, no zero row, and the rule is still one formula
            # for all classes. (At s = 1 exactly this degenerates to mode=level, zero rows included,
            # which is why s >= 1 is rejected below rather than silently allowed.)
            #
            # Offline on the real iNat prototypes (LEVEL=genus), cos-to-global / top5conf:
            #   s=0.00 1.0000/0.6399 (=global)   s=0.50 0.9763/0.5646   s=0.75 0.8972/0.4856
            #   s=0.90 0.7614/0.4502   s~0.92 ~0.74/~0.45 (the 0.72-0.75 winning band)
            #   s=0.95 0.6873/0.4495   s=1.00 0.2302/0.2624 (=level, 3000 zero rows)
            s = float(getattr(self.cfg, "PROMPT_CENTER_S", 0.92))
            lvs = [x.strip() for x in str(getattr(self.cfg, "PROMPT_CENTER_LEVEL", "genus")).split(",") if x.strip()]
            if not (0.0 <= s < 1.0):
                raise ValueError(f"PROMPT_CENTER_S must satisfy 0 <= s < 1 (got {s}); "
                                 "s=1 is PROMPT_CENTER_MODE=level, which produces zero rows")
            if not lvs:
                raise ValueError("PROMPT_CENTER_LEVEL is empty")
            mu_g = X.mean(0).unsqueeze(0).expand_as(X)
            # PROMPT_CENTER_LEVEL may list several levels; the weight s is split evenly over them, so
            # the subtracted vector stays a proper weighted average and the coefficients still sum to 1.
            # This is NOT cosmetic: with one level every level EFFECT is shrunk by the same factor
            # (a flat keep-profile), while with several the profile becomes a staircase -- e.g.
            # "family,genus" at s=0.92 keeps 0.080 of the kingdom..family effects but 0.540 of the
            # genus effect. Because C_j is a tail sum of non-negative weights the profile is always
            # monotone (fine effects are kept at least as much as coarse ones), and e_species is kept
            # in full for every choice, so this family only varies how much SHARED structure is stripped.
            # Measured on the real prototypes: listing all six levels is per-class cos 0.9997 to
            # mode=sum_all, i.e. the same arm; "family,genus" is the one combination that is genuinely
            # off every curve already run (<= 0.94 to all of them).
            mu_l = torch.zeros_like(X)
            for lv in lvs:
                if lv == "global":
                    mu_l = mu_l + mu_g
                else:
                    taxo = self._load_taxonomy()
                    if taxo is None:
                        raise ValueError("PROMPT_CENTER_MODE=blend with a taxonomy LEVEL needs categories.json")
                    mu_l = mu_l + self._level_mean(X, lv, taxo)
            mu_l = mu_l / float(len(lvs))
            out = X - (1.0 - s) * mu_g - s * mu_l
            nrm = out.norm(dim=-1)
            print(f"[PROMPT_CENTER blend] s={s} levels={','.join(lvs)} -> pre-norm row norm "
                  f"mean={nrm.mean().item():.3f} min={nrm.min().item():.3f} max={nrm.max().item():.3f}; "
                  f"{int((nrm < 1e-6).sum())}/{X.shape[0]} rows are ZERO (must be 0 for s < 1)")
        elif mode == "shrink":                   # PARTIAL centering with NO global term:
            #     out = O - s * mu_LEVEL
            # This is the alpha axis of mode=level_keep generalized: s = 0.5 reproduces level_keep
            # exactly (2O - mu is a positive multiple of O - 0.5 mu; verified per-class cos 1.00000000),
            # s = 1 is mode=level. It is zero-row-safe for s < 1, which is the reason to want it.
            #
            # BUT READ THIS BEFORE USING IT. A class alone in its group has mu_LEVEL = O, so the
            # expression collapses to (1-s) * O -- after row normalization, the RAW uncentered
            # prototype. Such a class receives NO centering at any s. Measured on iNat with
            # LEVEL=genus, cosine to the raw prototype direction, singleton vs non-singleton:
            #     s=0.50  1.0000 / 0.9747      s=0.90  1.0000 / 0.6196
            #     s=0.80  1.0000 / 0.8105      s=0.98  1.0000 / 0.3464
            # The 3000 singleton classes (36.8%) sit at exactly 1.0000 for every s, so turning s up
            # does not strengthen the method uniformly -- it SPLITS the initialization in two. Compare
            # mode=blend, where the global term reaches every class (singleton 0.7198, the same value
            # the plain global arm gives them), or mode=sum_all / mode=global, which are essentially
            # homogeneous (singleton-vs-rest gap 0.002 and 0.006).
            #
            # WHY IT IS STILL WORTH RUNNING: whether that inhomogeneity actually hurts has never been
            # measured. s = 0.963 matches blend s=0.92 on the NON-singleton classes (cos-to-raw 0.4141
            # vs 0.4149) while leaving singletons untouched, so the pair is a controlled experiment on
            # one question alone: should a class with no relatives get global centering, or nothing?
            # The two inits are per-class cos 0.8851, i.e. genuinely different arms.
            s = float(getattr(self.cfg, "PROMPT_CENTER_S", 0.92))
            lvs = [x.strip() for x in str(getattr(self.cfg, "PROMPT_CENTER_LEVEL", "genus")).split(",") if x.strip()]
            if not (0.0 <= s < 1.0):
                raise ValueError(f"PROMPT_CENTER_S must satisfy 0 <= s < 1 (got {s}); "
                                 "s=1 is PROMPT_CENTER_MODE=level, which produces zero rows")
            if not lvs:
                raise ValueError("PROMPT_CENTER_LEVEL is empty")
            mu_l = torch.zeros_like(X)
            for lv in lvs:
                if lv == "global":
                    mu_l = mu_l + X.mean(0).unsqueeze(0).expand_as(X)
                else:
                    taxo = self._load_taxonomy()
                    if taxo is None:
                        raise ValueError("PROMPT_CENTER_MODE=shrink with a taxonomy LEVEL needs categories.json")
                    mu_l = mu_l + self._level_mean(X, lv, taxo)
            # PROMPT_CENTER_G adds an independent global term:  out = O - g*mu_global - s*mean(mu_LEVELs).
            # g = 0 is plain shrink. g = 1 is "sum_k (O - mu_global - s*mu_k)" written in closed form,
            # where the subtracted coefficients sum to 1 + s > 1 -- deliberate OVER-centering, which
            # pushes rows PAST the origin rather than merely towards it. Measured on the real iNat
            # prototypes with all six levels, count of classes whose init ends up NEGATIVELY correlated
            # with their own raw prototype: s=0.3 -> 0/8142, s=0.5 -> 130/8142, s=0.963 -> 6390/8142
            # (mean cos to raw -0.167). Past roughly s=0.5 the classifier row for a species points away
            # from that species. The log below reports this count so it cannot be missed.
            g = float(getattr(self.cfg, "PROMPT_CENTER_G", 0.0))
            out = X - s * (mu_l / float(len(lvs)))
            if g != 0.0:
                out = out - g * X.mean(0).unsqueeze(0).expand_as(X)
            nrm = out.norm(dim=-1)
            # how many classes came out pointing exactly where they started -- the inhomogeneity above,
            # surfaced in the log rather than left to be discovered from the accuracies
            cos_raw = (F.normalize(out, dim=-1) * F.normalize(X, dim=-1)).sum(-1)
            untouched = int((cos_raw > 0.9999).sum())
            flipped = int((cos_raw < 0).sum())
            print(f"[PROMPT_CENTER shrink] s={s} g={g} levels={','.join(lvs)} -> pre-norm row norm "
                  f"mean={nrm.mean().item():.3f} min={nrm.min().item():.3f} max={nrm.max().item():.3f}; "
                  f"{int((nrm < 1e-6).sum())}/{X.shape[0]} rows are ZERO; "
                  f"{untouched}/{X.shape[0]} rows are UNCENTERED (identical direction to raw O); "
                  f"{flipped}/{X.shape[0]} rows are FLIPPED (negative cosine to raw O -- over-centered)")
        elif mode == "proj":                     # LEAST-SQUARES SUBSPACE REMOVAL.
            # Every other combination mode fixes the weights on the level means and subtracts a
            # weighted average. This one SOLVES for them: find the coefficients that make the residual
            # as small as possible, i.e. project O onto the orthogonal complement of the subspace the
            # level means span.
            #     c_hat = argmin_c || O - sum_k c_k mu_k ||        out = O - sum_k c_hat_k mu_k
            # Zero free parameters beyond which levels enter the span, and the weights adapt per class.
            #
            # WHY THE GATE IS LOAD-BEARING HERE: if a class is alone in its group then mu_LEVEL = O, so
            # O lies exactly IN the span and the residual is exactly ZERO -- the failure that destroyed
            # mode=level (2579 zero rows measured with no gate). A level is therefore admitted to the
            # span only when that class's group has >= PROMPT_CENTER_GENUS_MIN members, which is not an
            # arbitrary cutoff at its natural value 2: a group of one carries no information about the
            # class beyond the class itself. Measured with the gate: 0 zero rows, min pre-norm row norm
            # 1.520, mean span dimension 6.56 of 7.
            #
            # Offline geometry on the real iNat prototypes (cos-to-global / top5conf):
            #   all 7 levels, gate 2   0.5271 / 0.3963   <- lowest top5conf measured in this project
            #   all 7 levels, gate 5   0.6966 / 0.4313
            #   genus excluded, gate 2 0.7323 / 0.4749
            # The two gate settings are per-class cos 0.7914 apart, i.e. genuinely different arms.
            lvs = [x.strip() for x in str(getattr(self.cfg, "PROMPT_CENTER_LEVEL",
                   "global,kingdom,phylum,class,order,family,genus")).split(",") if x.strip()]
            ms = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            if not lvs:
                raise ValueError("PROMPT_CENTER_LEVEL is empty")
            C_, D_ = X.shape
            Xd = X.double()
            mus, keeps = [], []
            for lv in lvs:
                if lv == "global":
                    mus.append(Xd.mean(0).unsqueeze(0).expand_as(Xd))
                    keeps.append(torch.full((C_,), float(C_), dtype=torch.float64, device=X.device))
                    continue
                taxo = self._load_taxonomy()
                if taxo is None:
                    raise ValueError("PROMPT_CENTER_MODE=proj with a taxonomy LEVEL needs categories.json")
                mu = torch.zeros_like(Xd)
                cnt = torch.zeros(C_, dtype=torch.float64, device=X.device)
                groups = {}
                for i, name in enumerate(self.classnames):
                    key = taxo.get(name, {}).get(lv)
                    if key is None:
                        raise ValueError(f"taxonomy level '{lv}' missing for class '{name}'")
                    groups.setdefault(key, []).append(i)
                for _, idxs in groups.items():
                    it = torch.as_tensor(idxs, device=X.device)
                    mu[it] = Xd[it].mean(0)
                    cnt[it] = float(len(idxs))
                mus.append(mu); keeps.append(cnt)
            B = torch.stack(mus, dim=1).clone()                        # [C, K, D]
            keep = torch.stack(keeps, dim=1) >= float(ms)              # [C, K]
            B[~keep] = 0.0                                             # a gated-out level leaves the span
            K = len(lvs)
            G = torch.einsum("cid,cjd->cij", B, B)
            # ridge scaled by each class's own Gram trace: the level means are nested and therefore
            # highly collinear, and a zeroed-out row would otherwise make G singular.
            # PROMPT_CENTER_PROJ_RIDGE: at its default 1e-8 this is numerical hygiene. Raised, it becomes
            # a SMOOTH replacement for the size gate -- a singleton level makes O lie in the span and the
            # coefficients blow up, and the ridge caps that without a hard cutoff. Measured with the gate
            # switched off (PROMPT_CENTER_GENUS_MIN 1), zero rows / min pre-norm row norm / cos-to-global:
            #   1e-3 -> 0 / 0.0042 / 0.5351    1e-2 -> 0 / 0.0420 / 0.5669
            #   1e-1 -> 0 / 0.3916 / 0.7254    0.5  -> 0 / 1.6918 / 0.9009
            # Below ~1e-2 the smallest rows are numerically indistinguishable from the degenerate case, so
            # use lambda >= 0.1 if the gate is off. lambda -> inf converges to no centering at all.
            ridge = float(getattr(self.cfg, "PROMPT_CENTER_PROJ_RIDGE", 1e-8))
            eye = torch.eye(K, dtype=torch.float64, device=X.device)
            G = G + ridge * (G.diagonal(dim1=1, dim2=2).sum(-1) / K).clamp_min(1e-12)[:, None, None] * eye
            coef = torch.linalg.solve(G, torch.einsum("cid,cd->ci", B, Xd))
            out = (Xd - torch.einsum("ci,cid->cd", coef, B)).to(X.dtype)
            nrm = out.norm(dim=-1)
            print(f"[PROMPT_CENTER proj] levels={','.join(lvs)} gate={ms} ridge={ridge:g} -> mean span dim="
                  f"{keep.double().sum(1).mean().item():.2f}/{K}; pre-norm row norm "
                  f"mean={nrm.mean().item():.3f} min={nrm.min().item():.3f} max={nrm.max().item():.3f}; "
                  f"{int((nrm < 1e-6).sum())}/{C_} rows are ZERO (must be 0; a nonzero count means the "
                  f"gate let a singleton level into the span)")
        elif mode == "pick":                     # HARD per-class selection by ALIGNMENT, not by size.
            # cascade also picks exactly one level per class, but it picks the DEEPEST level whose group
            # is big enough -- a rule about coverage. This picks the level whose mean is most aligned
            # with the prototype itself, i.e. the one that explains the most of it:
            #     k*(i) = argmax_k  cos(mu_k(i), O_i)   over levels whose group has >= GENUS_MIN members
            #     out_i = O_i - mu_{k*}(i)
            # Same size gate as mode=proj and for the same reason: a singleton group has mu = O, which
            # would score cos = 1, win the argmax every time, and zero the row out.
            # Measured on the real iNat prototypes with gate 2, which level each class ends up choosing:
            #   genus 4678 | family 1639 | order 575 | global 492 | kingdom 327 | class 245 | phylum 186
            # -- so it usually lands on genus, but 3464 classes prefer a coarser level than cascade would
            # have given them. zero rows 0, min pre-norm row norm 1.561, cos-to-global 0.5631,
            # top5conf 0.4170; per-class cos 0.9534 to mode=proj at gate 2, i.e. a distinct arm.
            lvs = [x.strip() for x in str(getattr(self.cfg, "PROMPT_CENTER_LEVEL",
                   "global,kingdom,phylum,class,order,family,genus")).split(",") if x.strip()]
            ms = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            if not lvs:
                raise ValueError("PROMPT_CENTER_LEVEL is empty")
            C_ = X.shape[0]
            mus, cnts = [], []
            for lv in lvs:
                if lv == "global":
                    mus.append(X.mean(0).unsqueeze(0).expand_as(X))
                    cnts.append(torch.full((C_,), float(C_), device=X.device))
                    continue
                taxo = self._load_taxonomy()
                if taxo is None:
                    raise ValueError("PROMPT_CENTER_MODE=pick with a taxonomy LEVEL needs categories.json")
                mu = torch.zeros_like(X)
                cnt = torch.zeros(C_, device=X.device)
                groups = {}
                for i, name in enumerate(self.classnames):
                    key = taxo.get(name, {}).get(lv)
                    if key is None:
                        raise ValueError(f"taxonomy level '{lv}' missing for class '{name}'")
                    groups.setdefault(key, []).append(i)
                for _, idxs in groups.items():
                    it = torch.as_tensor(idxs, device=X.device)
                    mu[it] = X[it].mean(0)
                    cnt[it] = float(len(idxs))
                mus.append(mu); cnts.append(cnt)
            Bm = torch.stack(mus, dim=1)                                   # [C, K, D]
            ok = torch.stack(cnts, dim=1) >= float(ms)                     # [C, K]
            align = (F.normalize(Bm, dim=-1) * F.normalize(X, dim=-1).unsqueeze(1)).sum(-1)  # [C, K]
            align = align.masked_fill(~ok, -2.0)
            if bool((align.max(1).values <= -2.0).any()):
                raise ValueError("mode=pick: some class has no level passing the gate; lower "
                                 "PROMPT_CENTER_GENUS_MIN or include 'global' in PROMPT_CENTER_LEVEL")
            pick = align.argmax(1)
            out = X - Bm[torch.arange(C_, device=X.device), pick]
            nrm = out.norm(dim=-1)
            census = " ".join(f"{lv}={int((pick == i).sum())}" for i, lv in enumerate(lvs))
            print(f"[PROMPT_CENTER pick] levels={','.join(lvs)} gate={ms} -> chosen level: {census}"
                  f" | pre-norm row norm mean={nrm.mean().item():.3f} min={nrm.min().item():.3f} "
                  f"max={nrm.max().item():.3f}; {int((nrm < 1e-6).sum())}/{C_} rows are ZERO")
        elif mode == "sum_all":                  # ADD UP every level residual, no weights, no knobs:
            #     out = sum_{k} r_k,  r_k = O - mu_k  over k = global, kingdom, phylum, class, order,
            #                                             family, genus   (7 terms)
            # Row normalization removes the overall scale, so this is NOT a weakened centering: it
            # equals 7 * (O - mean_k mu_k), i.e. FULL-strength centering against the arithmetic mean of
            # the seven level means. In decomposition terms (verified to 1.4e-14 on the real
            # prototypes) it is
            #     sum_k r_k = 7 * ( sum_{j=1..6} (j/7) e_j + e_species )
            # so it keeps each level effect with a LINEAR ramp: kingdom 1/7, phylum 2/7, class 3/7,
            # order 4/7, family 5/7, genus 6/7, species 7/7. Coarse structure is removed most, fine
            # structure least. Zero free parameters.
            #
            # HONEST EXPECTATION: measured cos-to-global 0.9687 / top5conf 0.5792, i.e. geometrically a
            # near-duplicate of mode=global (1.0000/0.6399) and outside the 0.72-0.75 winning band. The
            # reason is visible in the ramp: the genus effect carries 46.4% of the prototype norm and
            # this keeps 6/7 of it. Predict a tie with global (80.52). It is here because it is the one
            # arm in this family with NO constant to justify.
            taxo = self._load_taxonomy()
            if taxo is None:
                raise ValueError("PROMPT_CENTER_MODE=sum_all needs a dataset with categories.json")
            LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus"]
            mu_total = X.mean(0).unsqueeze(0).expand_as(X).clone()        # mu_global
            for lv in LEVELS:
                mu_total = mu_total + self._level_mean(X, lv, taxo)
            out = float(len(LEVELS) + 1) * X - mu_total                   # == sum_k (X - mu_k)
            nrm = out.norm(dim=-1)
            print(f"[PROMPT_CENTER sum_all] levels=global+{LEVELS} (no knobs) -> pre-norm row norm "
                  f"mean={nrm.mean().item():.3f} min={nrm.min().item():.3f} max={nrm.max().item():.3f}; "
                  f"{int((nrm < 1e-6).sum())}/{X.shape[0]} rows are ZERO")
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
        elif mode == "hcluster":                 # C: taxonomy-free HIERARCHICAL cascade. Cuts ONE
            # agglomerative dendrogram at several granularities (finest -> coarsest) instead of reading
            # genus/family/order from a taxonomy file, then reuses cascade's exact fallback logic
            # (residual mean, min_size gate). Levels are GUARANTEED nested -- a fine cluster never
            # straddles two coarse clusters -- because every cut comes from the SAME linkage tree, the
            # same nesting property genus/family/order has by construction (unlike independent per-level
            # KMeans calls in 'cluster', which have no such guarantee).
            # Linkage = "complete" (max pairwise distance between two clusters), not "average": measured
            # offline that "average" linkage chains -- one cluster absorbed 4549/8142 classes at k=509,
            # a "rich get richer" blob (median cluster size only 3) that is barely different from plain
            # global centering for the majority of classes it nominally "covers". "complete" resists this
            # (same k=509: max cluster size 893, median 6) and reaches HIGHER 3-level coverage (8108/8142
            # = 99.6%, 34 fall to global) than "average" did (8100/8142, 42 fall to global).
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist
            # yacs literal_eval's CLI values, so "16,64,256" arrives as a tuple, "16" as a bare int,
            # and a YAML string stays a str -- accept all three rather than assuming one.
            raw_sizes = getattr(self.cfg, "PROMPT_CENTER_HCLUSTER_SIZES", (16, 64, 256))
            if isinstance(raw_sizes, (int, float)):
                raw_sizes = [raw_sizes]
            elif isinstance(raw_sizes, str):
                raw_sizes = [s for s in raw_sizes.split(",") if s.strip()]
            sizes = sorted({int(s) for s in raw_sizes})
            if not sizes or min(sizes) < 1:
                raise ValueError(f"PROMPT_CENTER_HCLUSTER_SIZES must be positive ints, got {raw_sizes}")
            min_size = int(getattr(self.cfg, "PROMPT_CENTER_GENUS_MIN", 5))
            Xn = F.normalize(X, dim=-1).cpu().numpy()
            Z = linkage(pdist(Xn, metric="cosine"), method="complete")
            local_mu = X.mean(0).unsqueeze(0).repeat(X.shape[0], 1)
            assigned = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
            used = []
            for s in sizes:                      # smallest size = finest (most clusters) first
                k = max(1, min(round(X.shape[0] / s), X.shape[0]))
                labels = torch.as_tensor(fcluster(Z, t=k, criterion="maxclust"), device=X.device)
                n_lv = 0
                for c in labels.unique().tolist():
                    idxs = (labels == c).nonzero(as_tuple=True)[0]
                    idxs = idxs[~assigned[idxs]]         # residual: only still-unassigned members count
                    if idxs.numel() < min_size:
                        continue
                    local_mu[idxs] = X[idxs].mean(0)
                    assigned[idxs] = True
                    n_lv += int(idxs.numel())
                used.append(f"size{s}(k={k})={n_lv}")
            used.append(f"global={int((~assigned).sum())}")
            print(f"[PROMPT_CENTER hcluster] sizes={sizes} min_size={min_size} -> " + " ".join(used))
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
