import os
import torch
import torch.nn as nn
from tqdm import tqdm

from metrics.iou import compute_iou
from utils.path import ensure_dir


class Trainer:
    """
    Trainer for DpSSeg.
    Implements deep supervision training exactly as described.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        cfg,
        logger
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger

        self.device = torch.device(cfg["runtime"]["device"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["training"]["lr"]
        )

        self.aux_weight = cfg["training"]["aux_weight"]

        self.use_amp = cfg["runtime"]["amp"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.best_miou = 0.0

    def compute_loss(self, main_out, aux_outs, target):

        main_loss = self.criterion(main_out, target)

        aux_loss = 0.0
        for aux in aux_outs:
            aux_loss += self.criterion(aux, target)

        aux_loss /= len(aux_outs)

        total_loss = main_loss + self.aux_weight * aux_loss

        return total_loss

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0.0

        pbar = tqdm(self.train_loader)

        for images, masks in pbar:

            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                main_out, aux_outs = self.model(images)
                loss = self.compute_loss(main_out, aux_outs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            pbar.set_description(f"Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_iou = 0.0

        for images, masks in self.val_loader:

            images = images.to(self.device)
            masks = masks.to(self.device)

            main_out, _ = self.model(images)

            miou = compute_iou(main_out, masks)

            total_iou += miou.item()

        return total_iou / len(self.val_loader)

    def save_checkpoint(self, epoch, miou):

        output_dir = self.cfg["project"]["output_dir"]
        ensure_dir(output_dir)

        save_path = os.path.join(
            output_dir,
            "best_model.pth"
        )

        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "miou": miou
        }, save_path)

        self.logger.info(f"Saved best model at epoch {epoch}")

    def fit(self):

        epochs = self.cfg["training"]["epochs"]

        for epoch in range(epochs):

            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            train_loss = self.train_one_epoch()
            val_miou = self.validate()

            self.logger.info(
                f"Train Loss: {train_loss:.4f} | Val mIoU: {val_miou:.4f}"
            )

            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, val_miou)

        self.logger.info("Training complete.")