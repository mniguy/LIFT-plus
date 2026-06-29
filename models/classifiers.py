import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Linear):
    def __init__(self, feat_dim, num_classes, bias=True, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
    
    def reset_parameters(self):
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class CosineClassifier(LinearClassifier):
    def __init__(self, feat_dim, num_classes, scale=30, bias=False, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
        self.scale = scale

    def forward(self, x):
        return F.linear(self.scale * F.normalize(x), F.normalize(self.weight), self.bias)


class CosineClassifierPCT(LinearClassifier):
    """Cosine classifier with a per-class learnable temperature (logit scale).

    Generalizes CosineClassifier's single global `scale` to a learnable per-class
    scale s_c (init to `scale`). Motivated by the finding that the global cosine
    scale (25->30) shifted tail accuracy more than any text-prior trick; letting
    head/tail classes use different effective margins is the natural extension.
    The s_c live inside the classifier module, so add_classifier registers them in
    the tuner and they are optimized like any other tuned parameter.
    """
    def __init__(self, feat_dim, num_classes, scale=30, bias=False, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
        self.scale = nn.Parameter(torch.full((num_classes,), float(scale),
                                             dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, x):
        cos = F.linear(F.normalize(x), F.normalize(self.weight))  # [B, C] cosine similarity
        logit = cos * self.scale                                  # per-class scale (broadcasts over batch)
        if self.bias is not None:
            logit = logit + self.bias
        return logit


class L2NormClassifier(LinearClassifier):
    def __init__(self, feat_dim, num_classes, bias=False, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
    
    def forward(self, x):
        return F.linear(x, F.normalize(self.weight), self.bias)


class LayerNormClassifier(LinearClassifier):
    def __init__(self, feat_dim, num_classes, bias=False, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype, device=device)

    def forward(self, x):
        return F.linear(self.ln(x), F.normalize(self.weight), self.bias)
