from typing import Iterator

from tqdm import tqdm

from modules.bert.pretraining import BertForPreTraining
from settings import PreTrainingSettings
from trainers.base import BaseTrainer


class PreTrainer(BaseTrainer[BertForPreTraining, PreTrainingSettings]):

    @property
    def total_steps(self) -> int:
        return self.settings.total_steps

    def _get_infinite_dataloader(self) -> Iterator:
        while True:
            for batch in self.train_loader:
                yield batch

    def set_progress_bar(self, desc: str | None = None) -> tqdm:
        return tqdm(range(self.total_steps), desc="Pretraining")  # type: ignore

    def train(self):
        self.model.train()
        train_iterator = self._get_infinite_dataloader()
        running_loss = 0.0
        for step in self.progress_bar:
            batch = next(train_iterator)
            loss_val = self._training_step(batch)
            running_loss += loss_val
            if (step + 1) % self.settings.log_interval_steps == 0:
                self._log_progress(running_loss, self.settings.log_interval_steps)
                running_loss = 0.0
            if (step + 1) % self.settings.save_interval_steps == 0:
                self._handle_checkpoint(step, loss_val)

    def _handle_checkpoint(self, step: int, current_loss: float):
        state = self._get_base_state_dict(
            {
                "step": step,
                "loss": current_loss,
            }
        )

        self.save_checkpoint("last.pt", state)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_checkpoint("best.pt", state)
