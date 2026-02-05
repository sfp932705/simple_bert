from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.base import T_DataItem
from modules.trainable import TrainableModel
from scheduler import BertScheduler
from settings import TrainingSettings

T_Setting = TypeVar("T_Setting", bound=TrainingSettings)
T_Model = TypeVar("T_Model", bound=TrainableModel)


class BaseTrainer(ABC, Generic[T_Model, T_Setting]):
    def __init__(
        self,
        settings: T_Setting,
        model: T_Model,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: BertScheduler | None = None,
    ):
        self.settings = settings
        self.best_loss = float("inf")
        self.best_accuracy = 0.0
        self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.settings.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=self.settings.learning_rate,
            eps=self.settings.adam_epsilon,
            weight_decay=self.settings.weight_decay,
        )
        self.scheduler = scheduler or BertScheduler(
            self.optimizer,
            num_warmup_steps=self.settings.warmup_steps,
            num_training_steps=self.total_steps,
        )
        self.progress_bar = self.set_progress_bar()

    @property
    @abstractmethod
    def total_steps(self) -> int:
        pass

    @abstractmethod
    def set_progress_bar(self, desc: str | None = None) -> tqdm:
        pass

    @abstractmethod
    def train(self):
        pass

    def _training_step(self, batch: T_DataItem) -> float:
        self.optimizer.zero_grad()
        loss = self.model.compute_loss_from_dataset_batch(batch.to(self.device))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def _get_base_state_dict(self, additional_keys: dict | None = None) -> dict:
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if additional_keys:
            state.update(additional_keys)
        return state

    def save_checkpoint(self, name: str, state: dict):
        path = self.settings.checkpoint_dir / name
        torch.save(state, path)

    def _log_progress(self, running_loss: float, steps: int):
        avg_loss = running_loss / steps
        current_lr = self.scheduler.get_last_lr()[0]
        postfix = {"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"}
        self.progress_bar.set_postfix(postfix)
