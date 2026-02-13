import torch
from tqdm import tqdm

from modules.bert.finetuning import BertForSequenceClassification
from settings import FinetuningSettings
from trainers.base import BaseTrainer


class FinetuningTrainer(BaseTrainer[BertForSequenceClassification, FinetuningSettings]):
    @property
    def total_steps(self) -> int:
        return self.settings.num_train_epochs * len(self.train_loader)

    def train(self):
        self.model.train()

        for epoch in range(self.settings.num_train_epochs):
            self.tracker.start_progress(
                total_steps=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.settings.num_train_epochs}",
            )
            self._run_train_epoch()
            if self.val_loader:
                val_acc, val_loss = self.evaluate()
                self.tracker.log_metrics(
                    {"Val/Accuracy": val_acc, "Val/Loss": val_loss}
                )
                self._handle_checkpoint(epoch, val_acc, val_loss)
        self.tracker.close()

    def _run_train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            loss_val = self._training_step(batch)
            running_loss += loss_val
            self.tracker.update_progress(step_increment=1, postfix={"loss": loss_val})
        self.tracker.log_metric("Train/Loss", running_loss / len(self.train_loader))

    def evaluate(self) -> tuple[float, float]:
        assert self.val_loader
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                logits = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    token_type_ids=batch.token_type_ids,
                )
                loss = self.model.compute_loss(logits, batch.labels)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == batch.labels).sum().item()
                total_samples += batch.labels.size(0)
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(self.val_loader)
        return accuracy, avg_loss

    def _handle_checkpoint(self, epoch: int, accuracy: float, loss: float):
        state = self._get_base_state_dict(
            {"epoch": epoch, "accuracy": accuracy, "loss": loss}
        )
        self.save_checkpoint("last.pt", state)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.save_checkpoint("best.pt", state)
