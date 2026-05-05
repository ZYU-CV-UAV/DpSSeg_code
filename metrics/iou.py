import torch


def compute_iou(pred, target, num_classes=2):
    """
    Compute mean IoU for segmentation.
    """

    pred = torch.argmax(pred, dim=1)

    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(torch.tensor(1.0, device=pred.device))
        else:
            ious.append(intersection / union)

    return torch.mean(torch.stack(ious))


def compute_tiou(pred, target, threshold=0.5):
    """
    Compute the TGS segmentation score (TIoU).
    """

    if pred.dim() == 4 and pred.size(1) == 2:
        pred_mask = torch.argmax(pred, dim=1)
    elif pred.dim() == 4 and pred.size(1) == 1:
        pred_mask = (pred[:, 0] > threshold).long()
    elif pred.dim() == 3:
        pred_mask = (pred > threshold).long() if pred.dtype.is_floating_point else pred.long()
    else:
        raise ValueError(f"Unsupported prediction shape: {pred.shape}")

    if target.dim() == 4 and target.size(1) == 1:
        target_mask = target[:, 0].long()
    elif target.dim() == 3:
        target_mask = target.long()
    else:
        raise ValueError(f"Unsupported target shape: {target.shape}")

    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()

    intersection = (pred_mask & target_mask).flatten(1).sum(dim=1).float()
    union = (pred_mask | target_mask).flatten(1).sum(dim=1).float()

    iou = torch.where(
        union > 0,
        intersection / union,
        torch.ones_like(union)
    )

    thresholds = torch.arange(
        0.50, 1.00, 0.05,
        device=pred.device,
        dtype=iou.dtype
    )

    tiou_per_image = (iou[:, None] > thresholds[None, :]).float().mean(dim=1)

    return tiou_per_image.mean()
