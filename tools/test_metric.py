import torch
from metrics.iou import compute_iou


def main():

    # Fake prediction (2 classes)
    pred = torch.randn(1, 2, 256, 256)

    # Fake target
    target = torch.randint(0, 2, (1, 256, 256))

    miou = compute_iou(pred, target)

    print("mIoU:", miou.item())


if __name__ == "__main__":
    main()