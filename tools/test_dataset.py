import yaml
import torch

from datasets.tgs_dataset import TGSDataset


def main():

    with open("configs/dpseg_tgs.yaml") as f:
        cfg = yaml.safe_load(f)

    dataset = TGSDataset(
        image_dir=cfg["data"]["train_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        img_size=cfg["data"]["img_size"]
    )

    print(f"Total samples: {len(dataset)}")

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image min/max:", image.min().item(), image.max().item())
    print("Mask unique values:", torch.unique(mask))


if __name__ == "__main__":
    main()