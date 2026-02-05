from typing import Iterator

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

    def train(self):
        self.model.train()
        train_iterator = self._get_infinite_dataloader()
        self.tracker.start_progress(self.total_steps, desc="Pretraining")
        running_loss = 0.0
        for step in range(self.total_steps):
            batch = next(train_iterator)
            loss_val = self._training_step(batch)
            running_loss += loss_val
            current_lr = self.scheduler.get_last_lr()[0]
            self.tracker.update_progress(
                step_increment=1, postfix={"loss": loss_val, "lr": current_lr}
            )
            if (step + 1) % self.settings.log_interval_steps == 0:
                avg_loss = running_loss / self.settings.log_interval_steps
                self.tracker.log_metric("Pretraining/Loss", avg_loss, step=step)
                self.tracker.log_metric("Pretraining/LR", current_lr, step=step)
                running_loss = 0.0
            last_step = (step + 1) == self.total_steps
            if (step + 1) % self.settings.save_interval_steps == 0 or last_step:
                self._handle_checkpoint(step, loss_val)
        self.tracker.close()

    def _handle_checkpoint(self, step: int, current_loss: float):
        state = self._get_base_state_dict({"step": step, "loss": current_loss})
        self.save_checkpoint("last.pt", state)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_checkpoint("best.pt", state)
