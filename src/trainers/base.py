from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch.optim import AdamW

from data.base import BaseDataset, T_DataItem
from modules.trainable import T_FPOutput, TrainableModel
from scheduler import BertScheduler
from settings import TrainingSettings
from tracker import ExperimentTracker

T_Setting = TypeVar("T_Setting", bound=TrainingSettings)
T_Model = TypeVar("T_Model", bound=TrainableModel)
T_Dataset = TypeVar("T_Dataset", bound=BaseDataset)


class BaseTrainer(ABC, Generic[T_Model, T_Setting]):
    def __init__(
        self,
        settings: T_Setting,
        model: T_Model,
        tracker: ExperimentTracker,
        train_dataset: T_Dataset,
        val_dataset: T_Dataset | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: BertScheduler | None = None,
    ):
        self.settings = settings
        self.tracker = tracker
        self.best_loss = float("inf")
        self.best_accuracy = float("-inf")
        self.device = torch.device(self.settings.device)
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.train_loader = train_dataset.loader()
        self.val_dataset = val_dataset
        self.val_loader = val_dataset.loader() if val_dataset else None
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
        self.model.to(self.device)

    @property
    @abstractmethod
    def total_steps(self) -> int:
        pass

    @abstractmethod
    def train(self):
        pass

    def _training_step(self, batch: T_DataItem) -> T_FPOutput:  # type: ignore
        self.optimizer.zero_grad()
        output = self.model.train_forward_from_dataset_batch(batch.to(self.device))
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        return output

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
        self.tracker.save_artifact(name, state)
