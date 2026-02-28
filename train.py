import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.dpseg import DpSSeg
from datasets.tgs_dataset import TGSDataset
from engine.trainer import Trainer
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.path import ensure_dir


def main():

    with open("configs/dpseg_tgs.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["runtime"]["seed"])

    ensure_dir(cfg["project"]["output_dir"])

    logger = setup_logger(
        cfg["project"]["output_dir"] + "/train.log"
    )

    dataset = TGSDataset(
        image_dir=cfg["data"]["train_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        img_size=cfg["data"]["img_size"]
    )

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        random_state=cfg["runtime"]["seed"]
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["runtime"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False
    )

    model = DpSSeg(num_classes=2)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        logger
    )

    trainer.fit()


if __name__ == "__main__":
    main()