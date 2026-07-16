import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    """ https://arxiv.org/abs/1708.02002
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, label):
        return focal_loss(F.cross_entropy(logit, label, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    """ https://arxiv.org/abs/1906.07413
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, logit, label, **kwargs):
        index = torch.zeros_like(logit, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        return F.cross_entropy(output, label, **kwargs)


class ClassBalancedLoss(nn.Module):
    """ https://arxiv.org/abs/1901.05555
    """
    def __init__(self, cls_num_list, beta=0.9999):
        super().__init__()
        per_cls_weights = (1.0 - beta) / (1.0 - (beta ** cls_num_list))
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, label, **kwargs):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, label, weight=self.per_cls_weights, **kwargs)


class GeneralizedReweightLoss(nn.Module):
    """ https://arxiv.org/abs/2103.16370
    """
    def __init__(self, cls_num_list, exp_scale=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        per_cls_weights = 1.0 / (cls_num_ratio ** exp_scale)
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, label, **kwargs):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, label, weight=self.per_cls_weights, **kwargs)


class BalancedSoftmaxLoss(nn.Module):
    """ https://arxiv.org/abs/2007.10740
    """
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

    def forward(self, logit, label, **kwargs):
        logit_adjusted = logit + self.log_cls_num
        return F.cross_entropy(logit_adjusted, label, **kwargs)


class LogitAdjustedLoss(nn.Module):
    """ https://arxiv.org/abs/2007.07314
    """
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit, label, **kwargs):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, label, **kwargs)
    
    
# class LogitKDLoss(nn.Module):
#     """
#     Knowledge-distillation style KL regularizer between teacher logits and student logits.
#     loss = KL( softmax(teacher/T) || softmax(student/T) ) * T^2
#     """
#     def __init__(self, T=1.0, reduction="batchmean"):
#         super().__init__()
#         self.T = float(T)
#         self.reduction = reduction

#     def forward(self, student_logits, teacher_logits):
#         T = self.T
#         # teacher는 고정 prior이므로 detach가 안전
#         p_t = F.softmax(teacher_logits / T, dim=1).detach()
#         log_p_s = F.log_softmax(student_logits / T, dim=1)
#         return F.kl_div(log_p_s, p_t, reduction=self.reduction) * (T * T)


class LogitKDLoss(nn.Module):
    """
    Knowledge-distillation style KL regularizer between teacher logits and student logits.
    L_KD = KL( softmax(z_t/T) || softmax(z_s/T) ) * T^2

    - reduction="none" 지원: per-sample KL([B]) 반환 → sample-wise λ 적용 가능
    """
    def __init__(self, T=1.0, reduction="batchmean"):
        super().__init__()
        self.T = float(T)
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits, reduction=None):
        T = self.T
        red = self.reduction if reduction is None else reduction

        # ✅ AMP 안정성: KD는 fp32로 계산
        s = student_logits.float()
        t = teacher_logits.float()

        p_t = F.softmax(t / T, dim=1).detach()
        log_p_s = F.log_softmax(s / T, dim=1)

        if red == "none":
            kl_mat = F.kl_div(log_p_s, p_t, reduction="none")  # [B, C]
            return kl_mat.sum(dim=1) * (T * T)                 # [B]

        return F.kl_div(log_p_s, p_t, reduction=red) * (T * T)

class InfoNCELoss(nn.Module):
    """
    InfoNCE (softmax contrastive) over class logits.
    L_i = -log softmax(logits_i / T)[y_i]

    - reduction="none" 지원: per-sample loss [B]
    """
    def __init__(self, T=0.1, reduction="mean"):
        super().__init__()
        self.T = float(T)
        self.reduction = reduction

    def forward(self, feat, text_proto, label, reduction=None):
        f      = F.normalize(feat.float(), dim=-1)        # fp32
        W      = F.normalize(text_proto.float(), dim=-1)  # fp32
        logits = (f @ W.t()) / self.T                     # [B, C]
        
        red = self.reduction if reduction is None else reduction
        return F.cross_entropy(logits, label, reduction=red)

class LADELoss(nn.Module):
    """ https://arxiv.org/abs/2012.00321
    """
    def __init__(self, cls_num_list, remine_lambda=0.1, estim_loss_weight=0.1):
        super().__init__()
        self.num_classes = len(cls_num_list)
        self.prior = cls_num_list / torch.sum(cls_num_list)

        self.balanced_prior = torch.tensor(1. / self.num_classes).float().to(self.prior.device)
        self.remine_lambda = remine_lambda

        self.cls_weight = cls_num_list / torch.sum(cls_num_list)
        self.estim_loss_weight = estim_loss_weight

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - math.log(N)
 
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, logit, label):
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss =  F.cross_entropy(logit_adjusted, label)

        per_cls_pred_spread = logit.T * (label == torch.arange(0, self.num_classes).view(-1, 1).type_as(label))  # C x N
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(label == torch.arange(0, self.num_classes).view(-1, 1).type_as(label), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        estim_loss = -torch.sum(estim_loss * self.cls_weight)

        return ce_loss + self.estim_loss_weight * estim_loss


class VSLoss(nn.Module):
    """ Vector-Scaling loss (Kini et al., NeurIPS 2021, https://arxiv.org/abs/2103.01550).
    Unifies multiplicative (CDT) and additive (LA/BS) logit adjustment:
        output_j = logit_j / Delta_j + iota_j
        Delta_j = (n_max / n_j)^gamma   (>= 1, head = 1; multiplicative -> recovers CDT at tau=0)
        iota_j  = tau * log(n_j / N)     (additive -> recovers LA at gamma=0)
    gamma=0 collapses to LA. Typical LT settings use gamma in [0.1, 0.3], tau in [1.0, 1.5].
    """
    def __init__(self, cls_num_list, gamma=0.3, tau=1.0):
        super().__init__()
        self.Delta = (cls_num_list.max() / cls_num_list) ** gamma
        self.iota = tau * torch.log(cls_num_list / cls_num_list.sum())

    def forward(self, logit, label, **kwargs):
        return F.cross_entropy(logit / self.Delta.unsqueeze(0) + self.iota.unsqueeze(0), label, **kwargs)
