# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LIFT+ is a research codebase implementing "LIFT+: Lightweight Fine-Tuning for Long-Tail Learning". It applies parameter-efficient fine-tuning (PEFT) methods (AdaptFormer, LoRA, Adapter, etc.) to CLIP and ViT backbones for long-tailed image classification on datasets like ImageNet-LT, Places-LT, iNaturalist 2018, and CIFAR-100-LT.

## Installation

```sh
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
conda install scikit-learn yacs tensorboard -c conda-forge
pip install openai-clip timm
```

Requires a single GPU with 24GB memory.

## Running Experiments

The main entry point is `main.py` with three required arguments:

```bash
python main.py -d [data] -b [backbone] -m [method] [options]
```

- `-d`: data config name (from `configs/data/`): `imagenet_lt`, `places_lt`, `inat2018`, `cifar100_ir100`, `cifar100_ir50`, `cifar100_ir10`
- `-b`: backbone config name (from `configs/backbone/`): `clip_vit_b16`, `in21k_vit_b16`, etc.
- `-m`: method config name (from `configs/method/`): `lift+`, `lift`, `zs`, `lp`, `fft`, `aft`, `coop`

**Quick start (auto-downloads CIFAR-100):**
```bash
python main.py -d cifar100_ir100 -b clip_vit_b16 -m lift+
```

**Key inline config overrides:**
```bash
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ root /path/to/datasets
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ gpu 0
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ accum_step 4   # gradient accumulation
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True  # test saved checkpoint
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ output_dir MyRun
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15
```

Output is saved to `output/<data>_<backbone>_<method>[_opts]/`.

## Architecture

**Config system** (`utils/config.py`): Uses `yacs.CfgNode`. Three YAML configs are merged (data → backbone → method), then CLI opts are applied. All options live in `_C` and are accessible via `cfg.*`. PEFT settings for the visual encoder are under `cfg.v.*`; for the text encoder under `cfg.l.*`.

**Model stack** (`models/`):
- `PEFT_Model` wraps a CLIP or TIMM model. All trainable parameters are collected into a `self.tuner` `nn.ParameterDict` — only params inside `tuner` are optimized.
- `PEFT_ViT` / `PEFT_RN` are image encoders; `PEFT_Text` is the text encoder.
- `PEFT_Block` wraps a transformer block and exposes `add_lora`, `add_adapter`, `add_adaptformer`, `add_ssf`, `add_aft` methods.
- Classifiers (`models/classifiers.py`): `CosineClassifier` (default for LIFT+), `LinearClassifier`, `L2NormClassifier`, `LayerNormClassifier`.
- Custom modules (`models/modules.py`): `LoRA`, `Adapter`, `AdaptFormer`, `SSF`, `MaskedLinear`, `NonHalfLayerNorm`.

**Trainer** (`trainer.py`): The `Trainer` class handles model building, data loading, loss, optimizer, training loop, and evaluation. PEFT modules are attached to the model during `build_tuner()`. The `warmup_peft()` method optionally pre-warms adapters before the main training loop.

**Loss functions** (`utils/losses.py`): `LogitAdjustedLoss` (default `LA`), `LDAMLoss`, `FocalLoss`, `BalancedSoftmaxLoss`, `ClassBalancedLoss`, `GeneralizedReweightLoss`, `LADELoss`, `InfoNCELoss`, `LogitKDLoss`. Loss is selected via `cfg.loss_type`.

**Datasets** (`datasets/`): `ImageNetLT`, `PlacesLT`, `iNat2018`, `CIFAR100LT`. All extend a common `_LTData` base. Dataset paths are configured in the corresponding `configs/data/*.yaml` files.

**Auxiliary losses**: Two regularization terms are available on top of the main classification loss:
- `TEXT_REG_LAMBDA` / `TEXT_REG_T`: KD-style regularization toward frozen text prototypes (logit KD).
- `INFONCE_LAMBDA` / `INFONCE_T`: InfoNCE contrastive loss between image features and text prototypes.

**Classifier initialization** (`classifier_init`): `"semantic"` initializes the classifier from CLIP text embeddings; `"hybrid"` uses a blend of text embeddings and wiki-caption embeddings (controlled by `SIM_THRESHOLD`, `HYBRID_TOPK`, `HYBRID_CAPTION_SOURCE`).

**MDA** (Minimalist Data Augmentation): When `mda: True`, crop scale increases progressively across epochs following a schedule (`mda_func`).

**TTE** (Test-Time Ensembling): When `tte: True`, uses FiveCrop and averages predictions.
