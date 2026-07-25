import torch
import torch.nn.functional as F


def MSELoss(inputs, masks, labels):
    inputs = torch.sigmoid(inputs)
    masks = 0.6 * (masks - 0.5) + 0.5  # 标签平滑
    weight =  (1 - labels).unsqueeze(2).unsqueeze(3)
    mse_loss = F.mse_loss(inputs, masks, reduction="none")
    mse_loss = (weight * mse_loss).mean(dim=(1, 2, 3))

    return mse_loss.sum() / max(weight.sum(), 1)


def BCELoss(inputs, labels):
    labels = 0.6 * (labels - 0.5) + 0.5  # 标签平滑
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, labels)

    return BCE_loss


def FocalLoss(inputs, labels):
    labels = 0.6 * (labels - 0.5) + 0.5  # 标签平滑
    alpha = 0.75
    gamma = 2

    BCE_loss = F.binary_cross_entropy_with_logits(inputs, labels, reduction="none")

    at = labels * alpha + (1 - labels) * (1 - alpha)
    pt = torch.exp(-BCE_loss)
    F_loss = (1 - pt) ** gamma * BCE_loss

    F_loss = at * F_loss

    return F_loss.sum()


def Loss(segs, masks, inputs, labels):

    return BCELoss(inputs, labels)
