from __future__ import annotations

import logging

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.trainable import TrainableModel
from settings import TrainerSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: TrainableModel,
        train_loader: DataLoader,
        settings: TrainerSettings,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.settings = settings
        self.device = torch.device(self.settings.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=self.settings.learning_rate,
            eps=self.settings.adam_epsilon,
            weight_decay=self.settings.weight_decay,
        )

    def train(self):
        self.model.train()
        self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.settings.num_train_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()
                loss = self.model.compute_loss_from_dataset_batch(batch.to(self.device))  # type: ignore
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if step % self.settings.log_interval == 0:
                    progress_bar.set_postfix({"loss": loss.item()})
            avg_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
            self.save_checkpoint(epoch, avg_loss)

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint_path = self.settings.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_path,
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")
