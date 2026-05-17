"""Extract image features + text prototypes for visualization.

Usage:
    python scripts/viz/extract_features.py \
        --ckpt output/final_tte/warmup/ep2_lr_1e-4/checkpoint.pth.tar \
        --output_dir output/viz/ours \
        -d imagenet_lt -b clip_vit_b16 -m lift+ \
        PEFT_WARMUP True PEFT_WARMUP_EPOCHS 2 PEFT_WARMUP_LR 1e-4 tte True

This loads a checkpoint, runs the test set, and saves:
    - image_features.npy : [N, D] test image features (post-PEFT)
    - labels.npy         : [N]    ground-truth labels
    - text_prototypes.npy: [C, D] class text prototypes (text_prior_weight)
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import setup_cfg  # noqa: E402
from trainer import Trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.pth.tar")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save .npy files")
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-b", "--backbone", type=str, required=True)
    parser.add_argument("-m", "--method", type=str, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    cfg.defrost()
    cfg.output_dir = args.output_dir
    cfg.freeze() if False else None  # leave unfrozen for Trainer
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = Trainer(cfg)

    # Load checkpoint into the tuner
    ckpt = torch.load(args.ckpt, map_location=trainer.device, weights_only=True)
    trainer.tuner.load_state_dict(ckpt["tuner"], strict=False)
    trainer.tuner.eval()

    if cfg.prec_test == "fp16":
        trainer.model.half()

    feats, labels = [], []
    with torch.no_grad():
        for image, label in tqdm(trainer.test_loader, desc="Extract", ascii=True):
            image = image.to(trainer.device)
            # if TTE, take center crop only for clean features
            if image.dim() == 5:
                image = image[:, 0]
            f = trainer.model(image=image, return_feature=True)
            feats.append(f.float().cpu().numpy())
            labels.append(label.numpy())

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Text prototypes (already normalized to unit norm)
    text_proto = trainer.text_prior_weight.float().cpu().numpy()

    np.save(os.path.join(args.output_dir, "image_features.npy"), feats)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)
    np.save(os.path.join(args.output_dir, "text_prototypes.npy"), text_proto)
    np.save(os.path.join(args.output_dir, "cls_num_list.npy"), np.asarray(trainer.cls_num_list))

    print(f"Saved features [{feats.shape}], labels [{labels.shape}], "
          f"text_proto [{text_proto.shape}] to {args.output_dir}")


if __name__ == "__main__":
    main()
