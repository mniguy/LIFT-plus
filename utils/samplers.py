from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler


def shuffle(a):
    a = torch.tensor(a)
    perm = torch.randperm(len(a))
    return a[perm].tolist()


class ClassBalancedSampler(Sampler):
    """Equal gradient share per class, at UNCHANGED total gradient budget.

    Draws the same number of samples per epoch as the dataset has (so steps/epoch, and therefore
    total compute, match the default shuffled loader exactly), but allocates those draws evenly
    across classes -- rare classes are repeated, frequent ones subsampled. Holding the budget
    fixed is what makes this a controlled intervention on B2 ("a class repairs the shared-direction
    bias in proportion to the gradient signal it receives") rather than a compute increase.

    Use with a loss that does NOT already correct for the class prior: under balanced sampling the
    effective training prior is uniform, so LA/BS/LDAM would double-correct. Pair it with CE.
    """

    def __init__(self, labels, num_samples=None):
        self.cls_idx_dict = defaultdict(list)
        for i, y in enumerate(labels):
            self.cls_idx_dict[y].append(i)
        self.classes = list(self.cls_idx_dict.keys())
        self.num_samples = num_samples if num_samples is not None else len(labels)

    def __iter__(self):
        n_cls = len(self.classes)
        per, rem = divmod(self.num_samples, n_cls)
        order = shuffle(list(range(n_cls)))          # which classes get the leftover draw
        extra = set(order[:rem])
        sampled_idx = []
        for k, c in enumerate(self.classes):
            n = per + (1 if k in extra else 0)
            pool = self.cls_idx_dict[c]
            if n <= len(pool):
                sampled_idx.extend(shuffle(pool)[:n])
            else:                                     # repeat with replacement for rare classes
                pick = torch.randint(len(pool), (n,)).tolist()
                sampled_idx.extend([pool[i] for i in pick])
        return iter(shuffle(sampled_idx))

    def __len__(self):
        return self.num_samples


class DownSampler(Sampler):
    def __init__(self, labels, n_max=100):
        self.cls_idx_dict = defaultdict(list)
        for i, y in enumerate(labels):
            self.cls_idx_dict[y].append(i)
        
        self.n_max = n_max
        self.cls_num_list = [min(n_max, len(cls_idx)) for cls_idx in self.cls_idx_dict.values()]
        self.num_samples = sum(self.cls_num_list)

    def __iter__(self):
        sampled_idx = []
        for cls_num, cls_idx in zip(self.cls_num_list, self.cls_idx_dict.values()):
            idx = shuffle(cls_idx)[:cls_num]
            sampled_idx.extend(idx)
        sampled_idx = shuffle(sampled_idx)
        
        for i in range(self.num_samples):
            yield sampled_idx[i]

    def __len__(self):
        return self.num_samples
