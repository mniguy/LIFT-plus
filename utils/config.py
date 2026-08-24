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
_C.train_sampler = "default"  # default (shuffled) | balanced (equal gradient share per class, same steps/epoch; pair with loss_type=CE)
_C.lr = 0.02
_C.weight_decay = 5e-4
_C.momentum = 0.9
_C.loss_type = "LA"  # Loss type (in utils/losses.py): CE Focal LDAM CB GRW BS LA LADE VS
_C.VS_GAMMA = 0.3    # VS loss: multiplicative (CDT) strength; 0 -> collapses to LA
_C.VS_TAU = 1.0      # VS loss: additive (LA) strength

_C.mda = True  # Minimalist data augmentation.
_C.mda_func = "convex"  # "min" / "convex" / "linear" / "concave" / "max".
_C.tte = False  # Test-time ensembling.
_C.SAVE_LOGITS = False  # If True, test() dumps raw logits.npy (float16) for offline margin analysis.
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
_C.classifier_init = "semantic"  # Classifier initialization method (semantic|class_mean|img_shrink|linear_probing).
_C.IMG_SHRINK_KAPPA = 20          # img_shrink init: lam_c = n_c/(n_c+kappa) blend of imagemean(head) vs centered-text(tail). int (yacs-strict); trainer casts to float.
_C.FREEZE_CLASSIFIER = False  # If True, keep the classifier at its init value (do not train it).
_C.FREEZE_ENCODER = False     # H_E test: freeze PEFT/encoder, train ONLY the classifier (inverse of FREEZE_CLASSIFIER).
_C.EVAL_CENTER = False        # H_B test: at TEST time, de-anisotropize the TRAINED classifier weight (decision-time centering).

# --- prototype centering / de-anisotropization (main research direction) ---
_C.PROMPT_CENTER = False        # semantic init: de-anisotropize prototypes
_C.PROMPT_CENTER_MODE = "global"  # I: global | group | tail | kappa | logcount | genus | genus_lex | diff_init | cascade | cascade_lex | nested | level | level_keep | taxo_kernel | blend | shrink | sum_all | proj | pick | cluster | hcluster | knn | std | whiten | pca ; J-controls: randdir | headonly | fewonly | perclass_rand
_C.PROMPT_CENTER_GAMMA = 0.03   # for mode=taxo_kernel: per-level decay of the taxonomic kernel, w_ij = gamma^d(i,j).
                                # gamma<=0 selects the limit "mean of the nearest non-empty relatives" (no hyperparameter).
                                # yacs is type-strict: pass 0.0, not 0.
_C.PROMPT_CENTER_S = 0.92     # for mode=blend/shrink: out = O - (1-s)*mu_global - s*mu_LEVEL. s=0 is mode=global,
                              # s->1 approaches mode=level. Must be < 1 (s=1 IS mode=level, zero rows and all).
                              # yacs is type-strict: pass 0.0, not 0.
_C.PROMPT_CENTER_G = 0.0      # for mode=shrink: weight of an extra global term,
                              #   out = O - g*mu_global - s*mean(mu_LEVELs).
                              # g=0 (default) is plain shrink. g=1 makes the subtracted coefficients
                              # sum to 1+s > 1, i.e. deliberate OVER-centering -- see the trainer note.
                              # yacs is type-strict: pass 0.0, not 0.
_C.PROMPT_CENTER_LEVEL = "genus"   # for mode=level/level_keep/blend/shrink/proj (comma-separated list allowed for blend/shrink): which taxonomy level supplies the group mean. global | genus | family | order | class | phylum | kingdom
_C.PROMPT_CENTER_PCA_K = 1         # for mode=pca: # top principal components to remove (0 == global mean-only)
_C.PROMPT_CENTER_KAPPA = 20        # for mode=kappa: rarity_c = kappa/(n_c+kappa) (int, yacs-strict; trainer casts to float)
_C.PROMPT_CENTER_PROJ_RIDGE = 0.00000001  # for mode=proj: ridge on the normal equations, scaled by each
                              # class's own Gram trace. 1e-8 is numerical hygiene only. Raising it turns the
                              # hard size gate into smooth regularization: lambda -> 0 is the plain
                              # projection, lambda -> inf leaves the prototype untouched. Pair with
                              # PROMPT_CENTER_GENUS_MIN 1 to drop the gate entirely.
_C.PROMPT_CENTER_GENUS_MIN = 5     # for mode=genus/cascade/proj/pick: min group size to use its own local mean (else fall to the next level)
_C.PROMPT_CENTER_CASCADE = "genus,family,order"  # for mode=cascade: taxonomy levels tried deepest-first before global
_C.PROMPT_CENTER_CASCADE_MEAN = "residual"  # for mode=cascade: a fallback level's mean is over its still-unassigned members ("residual") or over the whole group incl. deeper-assigned ones ("full")
_C.PROMPT_CENTER_CASCADE_NOFALL = False  # for mode=cascade: classes that qualify at NO level get
                              # NO centering (raw O) instead of the global mean. Turns the final
                              # fallback off entirely; the census still reports how many land there.
_C.PROMPT_CENTER_CASCADE_GLOBAL_FIRST = False  # for mode=cascade: remove the global centroid before cascading. A PROVABLE NO-OP (the global term cancels out of every later group mean; verified per-class cosine 1.0000 vs plain cascade), kept as a null control that isolates run-to-run variation
_C.PROMPT_CENTER_CASCADE_GLOBAL_LAST = False  # for mode=cascade: remove the residual's global centroid AFTER cascading. Near no-op (leftover centroid norm 0.0031 vs original 0.8278; per-class cosine 0.999937 to plain cascade), kept as the companion null control to GLOBAL_FIRST
_C.PROMPT_CENTER_CLUSTER_K = 100   # for mode=cluster: # k-means clusters over the prototypes (taxonomy-free local groups)
_C.PROMPT_CENTER_CLUSTER_SIZE = 0  # for mode=cluster: target AVG classes per cluster; >0 overrides _K via k=round(C/size), matching granularity (not cluster count) across datasets of different C
_C.PROMPT_CENTER_KNN_K = 20        # for mode=knn: # nearest-neighbor classes whose mean is subtracted (taxonomy-free local group)
_C.PROMPT_CENTER_HCLUSTER_SIZES = (16, 64, 256)  # for mode=hcluster: target AVG classes per cluster, finest->coarsest; cuts ONE agglomerative dendrogram at k=round(C/size) for each, so levels nest like genus/family/order do. MUST be a tuple, not a str: yacs literal_eval's the CLI value, and "16,64,256" parses to a tuple, so a str default would raise a type mismatch (unlike the bare-word level lists below, which literal_eval rejects and so stay str)
_C.PROMPT_CENTER_NESTED_LEVELS = "order,family,genus"  # for mode=nested: taxonomy levels centered REPEATEDLY, in the order given -- that order is the direction ("order,family,genus"=top-down, "genus,family,order"=bottom-up)
_C.PROMPT_CENTER_NESTED_MEAN = "recompute"  # for mode=nested: each level's mean is taken on the current residual ("recompute", a hierarchical decomposition) or on the raw prototypes and summed ("static", the deliberate over-subtraction control)
_C.PROMPT_CENTER_NESTED_RENORM = False  # for mode=nested: row-renormalize after EVERY level, not just at the end. Breaks the telescoping identity that otherwise makes coarse-to-fine subtraction collapse to "subtract the finest group mean"

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

# Prompt template selection for building text prototypes
_C.PROMPT_MODE            = "default"  # default / bare / places_scene / places_place / places_ensemble

_C.num_classes = 1000

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
