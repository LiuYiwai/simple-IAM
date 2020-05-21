from typing import Optional

import torch.nn.functional as F
from torch import Tensor


def binary_cross_entropy_loss(
        input: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean') -> Tensor:
    """Binary cross entropy loss.
    """

    return F.binary_cross_entropy(input, target, weight, reduction=reduction)


def multilabel_soft_margin_loss(
        input: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        difficult_samples: bool = False) -> Tensor:
    """Multilabel soft margin loss.
    """

    if difficult_samples:
        # label 1: positive samples
        # label 0: difficult samples
        # label -1: negative samples
        gt_label = target.clone()
        gt_label[gt_label == 0] = 1
        gt_label[gt_label == -1] = 0
    else:
        gt_label = target

    return F.multilabel_soft_margin_loss(input, gt_label, weight, reduction=reduction)
