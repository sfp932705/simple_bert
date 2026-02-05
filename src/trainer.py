from __future__ import annotations

import logging
from typing import Iterator

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.trainable import TrainableModel
from scheduler import BertScheduler
from settings import TrainingSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: TrainableModel,
        train_loader: DataLoader,
        settings: TrainingSettings,
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

        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * self.settings.num_train_epochs

        self.scheduler = BertScheduler(
            self.optimizer,
            num_warmup_steps=self.settings.warmup_steps,
            num_training_steps=self.total_steps,
        )

        self.best_loss = float("inf")

    def _get_infinite_dataloader(self) -> Iterator:
        while True:
            for batch in self.train_loader:
                yield batch

    def train(self):
        self.model.train()
        self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting training for {self.total_steps} steps.")
        train_iterator = self._get_infinite_dataloader()
        progress_bar = tqdm(range(self.total_steps), desc="Training")
        running_loss = 0.0
        for step in progress_bar:
            # 1. Get next batch (automatically resets loader if needed)
            batch = next(train_iterator)

            # 2. Optimization Step
            self.optimizer.zero_grad()
            loss = self.model.compute_loss_from_dataset_batch(batch.to(self.device))  # type: ignore
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 3. Logging
            running_loss += loss.item()

            if step % self.settings.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss = running_loss / self.settings.log_interval
                progress_bar.set_postfix(
                    {"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"}
                )
                running_loss = 0.0

            # 4. Checkpointing (End of "Epoch" equivalent or specific interval)
            # We save at the end of every "epoch" worth of steps
            if (step + 1) % self.steps_per_epoch == 0:
                epoch_num = (step + 1) // self.steps_per_epoch
                avg_epoch_loss = loss.item()  # Simplified for step-based
                logger.info(f"Step {step+1} (Epoch {epoch_num}) complete.")
                self.save_checkpoints(step, avg_epoch_loss)

    def save_checkpoints(self, step: int, current_loss: float):
        state = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": current_loss,
        }
        last_path = self.settings.checkpoint_dir / "last.pt"
        torch.save(state, last_path)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = self.settings.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(
                f"New best model saved at step {step} with loss {self.best_loss:.4f}"
            )
