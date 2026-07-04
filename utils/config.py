from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = None  # Dataset name.
_C.root = None  # Directory where datasets are stored.
_C.backbone = None  # CLIP-RN50, CLIP-ViT-B/32, CLIP-ViT-B/16, etc.
_C.resolution = None  # Resolution of input image.
_C.mean = None  # Normalize images with mean and std.
_C.std = None  # Normalize images with mean and std.

_C.seed = 0  # Use manual seed.
_C.deterministic = True  # Output reproducible results.
_C.gpu = 0  # Specify the GPU id. Use DataParallel when it is None.
_C.num_workers = 10  # Number of processes for data loading.
_C.prec_train = "amp"  # Model precision during training. "fp16" / "fp32" / "amp".
_C.prec_test = "fp16"  # Model precision during test. "fp16" / "fp32".

_C.num_epochs = 5
_C.batch_size = 128
_C.accum_step = 1  # Gradient accumulation step. Must be a divisor of batch_size.
_C.lr = 0.02
_C.weight_decay = 5e-4
_C.momentum = 0.9
_C.loss_type = "LA"  # Loss type (in utils/losses.py).

_C.mda = True  # Minimalist data augmentation.
_C.mda_func = "convex"  # "min" / "convex" / "linear" / "concave" / "max".
_C.tte = False  # Test-time ensembling.
_C.SAVE_LOGITS = False  # If True, test() dumps raw logits.npy (float16) for test-agnostic / prior-shift analysis.
_C.expand = None  # Test-time expanded size.

_C.zero_shot = False  # Zero-shot CLIP.
_C.coop = False  # context optimization.
_C.coop_init = None  # None (random) / "photo".
_C.coop_ctx_len = 4  # Length of learnable contexts.
_C.coop_cls_pos = "end"  # Position of class names in the prompts. "front" / "middle" / "end".
_C.proj_tuning = False  # Fine-tuning the image and text projections.
_C.clip_adapter = False  # Add CLIP adapters.
_C.clip_adapter_dim = 4  # CLIP adapters hidden dimension.
_C.classifier = None  # Classifier type (in models/classifiers.py). Use text encoder set when it is None.
_C.classifier_scale = 25  # Logit scale for classifier. (default=25)
_C.classifier_init = "semantic"  # Classifier initialization method.

_C.v = CN()
_C.v.fft = False  # Full fine-tuning (FFT).
_C.v.fft_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.bitfit = False  # Bias-terms fine-tuning (BitFit).
_C.v.bitfit_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.pt = False  # Prompt fine-tuning (PT).
_C.v.pt_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.pt_len = None  # Prompt lengths. Automatically set when it is None.
_C.v.lora = False  # Low-Rank Adapter (LoRA).
_C.v.lora_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.lora_dim = None  # LoRA bottleneck dimension. Automatically set when it is None.
_C.v.adapter = False  # Adapter.
_C.v.adapter_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.adapter_dim = None  # Adapter bottleneck dimension. Automatically set when it is None.
_C.v.adaptformer = False  # AdaptFormer.
_C.v.adaptformer_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.adaptformer_dim = None  # AdaptFormer bottleneck dimension. Automatically set when it is None.
_C.v.ssf = False  # Scaling & Shifting (SSF).
_C.v.ssf_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.aft = False  # Arbitrary fine-tuning.
_C.v.aft_layers = None  # None (all layers) / int (the last k layers) / expression (e.g. "[1, 2]", "range(3)", etc).
_C.v.aft_ratio = None  # Fine-tuning ratio.
_C.v.aft_loc = "all"  # Location of arbitrary fine-tuning parameters. "attn" / "mlp" / "all".
_C.v.aft_seed = 0  # Manual seed for generating mask.

# Textual Prior
_C.TEXT_REG_LAMBDA        = 0.001
_C.TEXT_REG_T             = 0.01

# InfoNCE
_C.INFONCE_LAMBDA         = 0.005
_C.INFONCE_T              = 0.08

# Warmup
_C.PEFT_WARMUP            = False
_C.PEFT_WARMUP_EPOCHS     = 2
_C.PEFT_WARMUP_STEPS      = -1
_C.PEFT_WARMUP_LR         = 5e-4

_C.PEFT_WARMUP_IMAGE      = True
_C.PEFT_WARMUP_TEXT       = False
_C.PEFT_WARMUP_PROJ       = False
_C.PEFT_WARMUP_CLASSIFIER = False

_C.WARMUP_TEXT_REG_LAMBDA = 0.0001
_C.WARMUP_TEXT_REG_T      = 0.001
_C.WARMUP_INFONCE_LAMBDA  = 0.0
_C.WARMUP_INFONCE_T       = 0.1

# Hybrid Caption
_C.SIM_THRESHOLD          = 0.6
_C.CHUNK_SIZE             = 50

_C.HYBRID_TOPK            = 8
_C.HYBRID_CAPTION_SOURCE  = "wiki"

# Dataset-aware text prior
_C.PROMPT_MODE            = "default"  # default / places_scene / places_place / places_ensemble
_C.PRIOR_REG_MODE         = "fixed"    # fixed / class_gate
_C.PRIOR_GATE_SOURCE      = "image_text"  # image_text / frequency / shuffled — signal driving the per-class gate
_C.PRIOR_GATE_POWER       = 1.0
_C.PRIOR_GATE_NORM        = "minmax"   # none / minmax / rank — rescale raw per-class similarity to a usable [0,1] range
_C.PRIOR_GATE_INVERT      = False      # False: high-sim classes get stronger reg. True: low-sim classes get stronger reg (1-gate).

# Regularization (KD / InfoNCE) annealing over the main training loop:
# strong early (stabilize init), weak late (let LA fit the visual boundary).
_C.REG_ANNEAL             = "none"     # none / linear / cosine
_C.REG_ANNEAL_END         = 0.0        # final scale (fraction of base lambda) at the last epoch
_C.REG_ANNEAL_START_EPOCH = 0          # keep scale=1.0 before this epoch, then decay

_C.num_classes = 1000
_C.wiki_caption_dir = "datasets/ImageNet_LT/wiki"
_C.WIKI_MAX_SENTENCES = 0    # 0이면 전체 문장 사용
_C.WIKI_MAX_CHARS = 0        # 0이면 글자 수 제한 없음

_C.WEIGHTS_PATH = ""
_C.l = CN()
_C.l.fft = False
_C.l.fft_layers = None
_C.l.bitfit = False
_C.l.bitfit_layers = None
_C.l.pt = False
_C.l.pt_layers = "deep"
_C.l.pt_len = 2
_C.l.lora = False
_C.l.lora_layers = None
_C.l.lora_dim = 4
_C.l.adapter = False
_C.l.adapter_layers = None
_C.l.adapter_dim = 4
_C.l.adaptformer = False
_C.l.adaptformer_layers = None
_C.l.adaptformer_dim = 4
_C.l.ssf = False
_C.l.ssf_layers = None
_C.l.aft = False
_C.l.aft_layers = None
_C.l.aft_ratio = None
_C.l.aft_loc = "all"
_C.l.aft_seed = 0

_C.test_only = False  # Load model and test.
_C.model_dir = None  # Directory to save the model checkpoint.
_C.output_dir = None  # Directory to save the output files (like log.txt and model weights).
_C.print_freq = 10  # How often (batches) to print training information.
