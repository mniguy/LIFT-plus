# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LIFT+ is a research codebase implementing "LIFT+: Lightweight Fine-Tuning for Long-Tail Learning". It applies parameter-efficient fine-tuning (PEFT) methods (AdaptFormer, LoRA, Adapter, etc.) to CLIP and ViT backbones for long-tailed image classification on datasets like ImageNet-LT, Places-LT, iNaturalist 2018, and CIFAR-100-LT.

## Current state — read these before proposing work

| | |
|---|---|
| `docs/FINDINGS.md` | What the centering line has established, **and what has been ruled out**. Several obvious directions are already closed; a few numbers still written in `scripts/*.sh` headers are superseded. |
| `docs/results.tsv` | Every measured number, with the config that produced it. `output/` is gitignored, so this is the only durable record. Regenerate with `python scripts/dump_results.py`. |

**Read `docs/FINDINGS.md` before designing a centering experiment.** Noise floor is ~0.08 in All,
Head carries ±0.5 of noise and must not be interpreted, and every number so far is a single seed.

After an experiment set finishes: rerun `dump_results.py` and add a paragraph to `FINDINGS.md`.
`python scripts/dump_results.py --check` exits non-zero when the file is stale.

This file is also served to other agents as `AGENTS.md` (a symlink), so both toolchains read the
same instructions. Edit `CLAUDE.md`; never replace the symlink with a second copy.

## Installation

See `README.md`. Requires one 24 GB+ GPU.

## Running Experiments

The main entry point is `main.py` with three required arguments:

```bash
python main.py -d [data] -b [backbone] -m [method] [options]
```

The three names are filenames under `configs/data/`, `configs/backbone/`, and
`configs/method/`; they are merged in that order, then CLI overrides are applied.

**Quick start (auto-downloads CIFAR-100):**
```bash
python main.py -d cifar100_ir100 -b clip_vit_b16 -m lift+
```

**Key inline config overrides** (YACS `key value` pairs, after the named arguments):

```bash
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 seed 0 output_dir MyRun
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ accum_step 4      # gradient accumulation
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True    # evaluate a checkpoint
```

Output is saved to `output/<data>_<backbone>_<method>[_opts]/`.

## Architecture

**Config system** (`utils/config.py`): Uses `yacs.CfgNode`. Three YAML configs are merged (data → backbone → method), then CLI opts are applied. All options live in `_C` and are accessible via `cfg.*`. PEFT settings for the visual encoder are under `cfg.v.*`; for the text encoder under `cfg.l.*`.

**Model stack** (`models/`):
- `PEFT_Model` wraps a CLIP or TIMM model. All trainable parameters are collected into a `self.tuner` `nn.ParameterDict` — only params inside `tuner` are optimized.
- `PEFT_ViT` / `PEFT_RN` are image encoders; `PEFT_Text` is the text encoder.
- `PEFT_Block` wraps a transformer block and exposes `add_lora`, `add_adapter`, `add_adaptformer`, `add_ssf`, `add_aft` methods.
- Classifiers (`models/classifiers.py`): `CosineClassifier` (default for LIFT+), `CosineClassifierPCT`, `LinearClassifier`, `L2NormClassifier`, `LayerNormClassifier`.
- Custom modules (`models/modules.py`): `LoRA`, `Adapter`, `AdaptFormer`, `SSF`, `MaskedLinear`, `NonHalfLayerNorm`.

**Trainer** (`trainer.py`): The `Trainer` class handles model building, data loading, loss, optimizer, training loop, and evaluation. PEFT modules are attached to the model during `build_tuner()`.

**Loss functions** (`utils/losses.py`): `LogitAdjustedLoss` (default `LA`), `LDAMLoss`, `FocalLoss`, `BalancedSoftmaxLoss`, `ClassBalancedLoss`, `GeneralizedReweightLoss`, `LADELoss`, `VSLoss`. Loss is selected via `cfg.loss_type`.

**Datasets** (`datasets/`): `ImageNetLT`, `PlacesLT`, `iNat2018`, `CIFAR100LT`. All extend a common `_LTData` base. Dataset paths are configured in the corresponding `configs/data/*.yaml` files.

**Classifier initialization** (`classifier_init`): `"semantic"` (default) initializes the classifier from CLIP text embeddings; `"class_mean"` and `"linear_probing"` use train image features; `"img_shrink"` blends class-mean image features (head) with centered text prototypes (tail), controlled by `IMG_SHRINK_KAPPA`.

**Prototype centering** (the current research direction): `PROMPT_CENTER: True` de-anisotropizes the text prototypes used to initialize the classifier. `PROMPT_CENTER_MODE` selects what is subtracted (`global`, `group`, `kappa`, `pca`, `knn`, `genus`/`cascade` for iNat taxonomy, plus negative controls like `randdir`, `headonly`). `EVAL_CENTER: True` instead centers the *trained* classifier weight at test time. `FREEZE_CLASSIFIER` / `FREEZE_ENCODER` isolate which side the effect comes from.

**MDA** (Minimalist Data Augmentation): When `mda: True`, crop scale increases progressively across epochs following a schedule (`mda_func`).

**TTE** (Test-Time Ensembling): When `tte: True`, uses FiveCrop and averages predictions.

## Conventions

Style, naming, and commit hygiene follow the global rules and the surrounding code — nothing
repo-specific to state. What follows is only what reading the code will *not* tell you.

**Checks before committing.** No linter, formatter, or test suite exists:

```bash
python -m compileall main.py trainer.py models datasets utils scripts
bash -n scripts/<edited>.sh     # shell edits -- a conflict marker once shipped unnoticed
python scripts/dump_results.py --check
```

**Never edit a script while it is executing.** bash re-reads the file at its next byte offset, so
an in-place edit can make a running experiment misparse. Check first:
`ps -eo args | grep '[r]un_center'`.

**Experiment scripts carry their reasoning in the header** — measured numbers, gotchas, and what
was already ruled out. That is why they are long; keep the convention. **When a header's numbers
are superseded, correct them** rather than leaving both — an agent reading the script cold will
believe a stale number and propose a closed direction.

**Comparing runs.** Before comparing two arms, confirm their configs actually match — check
`docs/results.tsv`, not the directory names. `min_size`, `renorm`, and epoch count have all
silently differed between runs that looked comparable. When reporting a result, give the exact
data/backbone/method, seed, GPU, and metrics.

**GPU discipline.** One training job per card. Sharing measured 0.29 → 0.72 s/batch (2.4×) and
slowed the neighbour too; memory is never the constraint (17 GB of 49 GB), SM contention is.
Launch detached so runs survive a dropped SSH session:

```bash
setsid nohup bash scripts/<launcher>.sh > output/<root>/_launch/launcher.log 2>&1 < /dev/null &
```

`PYTHONNOUSERSITE=1` is load-bearing: `~/.local/lib/python3.11` shadows the `ltl` conda env
(`/home/mingyu/.conda/envs/ltl/bin/python`) with newer clip/timm/yacs/sklearn, and user-site
outranks the env.

Experiment scripts take `GPU_ID`, `ARMS`, `OUT_ROOT`, `SUFFIX`, `GPUS`, `DRY_RUN`, `FORCE`.
Nothing under `output/` is ever committed.
