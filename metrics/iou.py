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
            ious.append(torch.tensor(1.0))
        else:
            ious.append(intersection / union)

    return torch.mean(torch.stack(ious))