#!/usr/bin/env python
"""#3 Tail feature distribution calibration (Yang et al., ICLR'21 style), text-free.

On top of a trained LIFT+ checkpoint (frozen), in the image-feature space:
  1. extract per-class CLIP features (train + test, single-crop).
  2. model each class as a Gaussian; for tail classes borrow covariance from the
     k nearest head ("base") classes (transfer) + shrinkage.
  3. sample synthetic features so every class reaches `target_n` -> balanced set.
  4. RE-TRAIN only the cosine classifier head on the (real+synthetic) balanced set.
  5. evaluate the ORIGINAL head vs the CALIBRATED head on the SAME single-crop test
     features (controlled: same features, only the head differs).

No text prior anywhere. Heavy imports (Trainer/CLIP) are inside main() so the math
functions below can be unit-tested standalone.

Usage:
  python scripts/feature_calibration.py -d imagenet_lt -b clip_vit_b16 -m lift+ \
      --ckpt output/calib/imagenet_lt/lift+_train --out output/calib/imagenet_lt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------- calibration math -----------------------------
def class_means_counts(feats, labels, C):
    D = feats.size(1)
    means = torch.zeros(C, D)
    counts = torch.zeros(C, dtype=torch.long)
    for c in range(C):
        m = labels == c
        counts[c] = int(m.sum())
        if counts[c] > 0:
            means[c] = feats[m].mean(0)
    return means, counts


def base_covariances(feats, labels, base_classes):
    """Per-class covariance for base (head) classes; returns {class: [D,D]}."""
    covs = {}
    for c in base_classes:
        fc = feats[labels == c].numpy()
        covs[int(c)] = torch.from_numpy(np.cov(fc, rowvar=False)).float()
    return covs


def synthesize(means, counts, covs_base, base_classes, shared_cov, C, D,
               k, alpha, target_n):
    """Sample synthetic features so each class reaches target_n.
    Tail-class covariance = mean of k nearest base-class covs (+ alpha*I),
    or a single shared pooled cov if shared_cov is given."""
    base_means = means[base_classes]  # [B, D]
    eye = torch.eye(D)
    syn_f, syn_y = [], []
    for c in range(C):
        need = target_n - int(counts[c])
        if need <= 0:
            continue
        if shared_cov is not None:
            cov = shared_cov
        else:
            d = (base_means - means[c]).pow(2).sum(1)
            nn = [int(base_classes[i]) for i in torch.argsort(d)[:k]]
            cov = torch.stack([covs_base[i] for i in nn]).mean(0)
        cov = cov + alpha * eye
        L = torch.linalg.cholesky(cov + 1e-4 * eye)
        z = torch.randn(need, D)
        syn_f.append(means[c] + z @ L.T)
        syn_y.append(torch.full((need,), c, dtype=torch.long))
    if not syn_f:
        return torch.empty(0, D), torch.empty(0, dtype=torch.long)
    return torch.cat(syn_f), torch.cat(syn_y)


def build_balanced(feats, labels, syn_f, syn_y, C, target_n):
    """Cap each real class at target_n, add synthetic -> ~balanced training set."""
    F_, Y_ = [], []
    for c in range(C):
        rc = feats[labels == c]
        if len(rc) > target_n:
            rc = rc[torch.randperm(len(rc))[:target_n]]
        F_.append(rc)
        Y_.append(torch.full((len(rc),), c, dtype=torch.long))
    if len(syn_f):
        F_.append(syn_f)
        Y_.append(syn_y)
    return torch.cat(F_), torch.cat(Y_)


def train_head(feats, labels, C, scale, init_w, epochs, bs, lr, device):
    D = feats.size(1)
    W = torch.nn.Parameter((init_w.clone() if init_w is not None
                            else F.normalize(torch.randn(C, D), dim=1)).to(device))
    opt = torch.optim.SGD([W], lr=lr, momentum=0.9, weight_decay=5e-4)
    feats, labels = feats.to(device), labels.to(device)
    N = len(feats)
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            logit = scale * F.linear(F.normalize(feats[idx]), F.normalize(W))
            loss = F.cross_entropy(logit, labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return F.normalize(W.detach(), dim=1).cpu()


def eval_head(W, feats, labels, C, scale):
    logit = scale * F.linear(F.normalize(feats), F.normalize(W))
    pred = logit.argmax(1)
    rec = np.zeros(C)
    for c in range(C):
        m = labels == c
        if m.any():
            rec[c] = (pred[m] == c).float().mean().item() * 100
    return rec


def splits(cls_num):
    c = np.asarray(cls_num)
    return (c > 100), ((c >= 20) & (c <= 100)), (c < 20)


def report(name, rec, h, m, fw):
    v = np.array([rec.mean(), rec[h].mean(), rec[m].mean(), rec[fw].mean()])
    print(f"  {name:12s} all {v[0]:7.3f}  head {v[1]:7.3f}  med {v[2]:7.3f}  few {v[3]:7.3f}")
    return v


# --------------------------------- driver -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", "-d", default="imagenet_lt")
    ap.add_argument("--backbone", "-b", default="clip_vit_b16")
    ap.add_argument("--method", "-m", default="lift+")
    ap.add_argument("--ckpt", required=True, help="trained LIFT+ checkpoint dir (has checkpoint.pth.tar)")
    ap.add_argument("--out", default="output/calib")
    ap.add_argument("--k", type=int, default=2, help="nearest base classes for cov transfer")
    ap.add_argument("--alpha", type=float, default=0.21, help="cov shrinkage added to diagonal")
    ap.add_argument("--target-n", type=int, default=200, help="per-class sample count after balancing")
    ap.add_argument("--base-min", type=int, default=100, help="min train count to be a 'base' (reliable cov) class")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--shared-cov", action="store_true", help="use one pooled cov instead of k-NN transfer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.config import _C as cfg
    from trainer import Trainer

    cfg.defrost()
    cfg.merge_from_file(os.path.join("./configs/data", args.data + ".yaml"))
    cfg.merge_from_file(os.path.join("./configs/backbone", args.backbone + ".yaml"))
    cfg.merge_from_file(os.path.join("./configs/method", args.method + ".yaml"))
    # single-crop features; this is a test_only-style feature pass (no training-time text reg)
    cfg.merge_from_list(["classifier_init", "semantic", "test_only", "True",
                         "tte", "False", "TEXT_REG_LAMBDA", "0.0", "INFONCE_LAMBDA", "0.0",
                         "PRIOR_REG_MODE", "fixed"])
    if args.opts:
        cfg.merge_from_list([o for o in args.opts if o])
    cfg.output_dir = args.out
    os.makedirs(cfg.output_dir, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.load_model(args.ckpt)
    C = trainer.num_classes
    scale = float(getattr(cfg, "classifier_scale", 30))
    device = trainer.device

    trainer.model.eval()
    with torch.no_grad():
        tr_f, tr_y = trainer.compute_train_features()
        te_f, te_y = [], []
        from tqdm import tqdm
        for image, label in tqdm(trainer.test_loader, ascii=True, desc="Test feats"):
            te_f.append(trainer.model(image.to(device), return_feature=True).float().cpu())
            te_y.append(label.clone())
    tr_f, tr_y = tr_f.float().cpu(), tr_y.long().cpu()
    te_f, te_y = torch.cat(te_f), torch.cat(te_y).long()
    D = tr_f.size(1)
    cls_num = np.asarray(trainer.cls_num_list)
    h, m, fw = splits(cls_num)
    init_w = trainer.model.tuner.classifier.weight.detach().float().cpu()  # trained LIFT+ head

    means, counts = class_means_counts(tr_f, tr_y, C)
    base_classes = (counts >= args.base_min).nonzero().flatten()
    assert len(base_classes) >= args.k, f"only {len(base_classes)} base classes (>= base_min={args.base_min})"
    shared_cov = None
    if args.shared_cov:
        # pooled within-class covariance over base classes
        acc = torch.zeros(D, D); n = 0
        for c in base_classes.tolist():
            fc = tr_f[tr_y == c]
            shifted = (fc - fc.mean(0)).numpy()
            acc += torch.from_numpy(shifted.T @ shifted).float(); n += len(fc)
        shared_cov = acc / max(n - len(base_classes), 1)
        covs_base = None
    else:
        covs_base = base_covariances(tr_f, tr_y, base_classes.tolist())

    syn_f, syn_y = synthesize(means, counts, covs_base, base_classes, shared_cov,
                              C, D, args.k, args.alpha, args.target_n)
    bal_f, bal_y = build_balanced(tr_f, tr_y, syn_f, syn_y, C, args.target_n)
    print(f"[calib] base classes={len(base_classes)}  synth feats={len(syn_y)}  "
          f"balanced set={len(bal_y)}  (target_n={args.target_n}, k={args.k}, alpha={args.alpha}, "
          f"shared_cov={args.shared_cov})")

    W_cal = train_head(bal_f, bal_y, C, scale, init_w, args.epochs, args.bs, args.lr, device)

    print(f"\n=== feature calibration  {args.data}  (head={int(h.sum())} med={int(m.sum())} few={int(fw.sum())}, "
          f"single-crop test) ===")
    rec_base = eval_head(init_w, te_f, te_y, C, scale)
    rec_cal = eval_head(W_cal, te_f, te_y, C, scale)
    v_base = report("LIFT+ (orig head)", rec_base, h, m, fw)
    v_cal = report("calibrated", rec_cal, h, m, fw)
    d = v_cal - v_base
    print(f"  {'Δ vs orig':12s} all {d[0]:+7.3f}  head {d[1]:+7.3f}  med {d[2]:+7.3f}  few {d[3]:+7.3f}")

    np.save(os.path.join(cfg.output_dir, "cls_accs_orig.npy"), rec_base)
    np.save(os.path.join(cfg.output_dir, "cls_accs_calib.npy"), rec_cal)
    np.save(os.path.join(cfg.output_dir, "cls_num_list.npy"), cls_num)
    print(f"\n[save] {cfg.output_dir}/cls_accs_{{orig,calib}}.npy")


if __name__ == "__main__":
    main()
