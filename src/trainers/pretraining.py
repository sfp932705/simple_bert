from typing import Iterator

import torch

from data.types.outputs.pretraining import PretrainingOutput
from modules.bert.pretraining import BertForPreTraining, PretrainingForwardPassOutput
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
            forward_output: PretrainingForwardPassOutput = self._training_step(batch)
            loss_val = forward_output.loss.item()
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
                self._log_predictions(batch, forward_output, step)
            last_step = (step + 1) == self.total_steps
            if (step + 1) % self.settings.save_interval_steps == 0 or last_step:
                self._handle_checkpoint(step, loss_val)
        self.tracker.close()

    def _format_mlm_sequence(
        self,
        seq_ids: list[int],
        seq_preds: list[int],
        seq_trues: list[int],
        seq_masks: list[bool],
    ) -> str:
        tokenizer = self.train_dataset.tokenizer
        sentence_parts = []
        for j in range(len(seq_ids)):
            if seq_masks[j]:
                p_tok = tokenizer.inverse_vocab.get(seq_preds[j], tokenizer.unk_token)
                t_tok = tokenizer.inverse_vocab.get(seq_trues[j], tokenizer.unk_token)
                sentence_parts.append(f"**[{p_tok} / {t_tok}]**")
            else:
                base_tok = tokenizer.inverse_vocab.get(seq_ids[j], tokenizer.unk_token)
                sentence_parts.append(base_tok)
        return "".join(sentence_parts).replace(tokenizer.delimiter, " ").strip()

    def _format_nsp_prediction(self, is_next_pred: bool, is_next_true: bool) -> str:
        nsp_pred_str = "IsNext" if is_next_pred else "NotNext"
        nsp_true_str = "IsNext" if is_next_true else "NotNext"
        match_icon = "✅" if is_next_pred == is_next_true else "❌"
        return f"Pred: {nsp_pred_str} | True: {nsp_true_str} {match_icon}"

    def _log_predictions(
        self,
        batch: PretrainingOutput,
        preds: PretrainingForwardPassOutput,
        step: int,
        samples: int = 5,
    ):
        num_samples = min(samples, batch.input_ids.size(0))

        ids = batch.input_ids[:num_samples]
        labels = batch.mlm_labels[:num_samples]
        mlm_preds = torch.argmax(preds.mlm_preds[:num_samples], dim=-1)
        nsp_preds = torch.argmax(preds.nsp_preds[:num_samples], dim=-1)

        safe_labels = torch.where(labels != -100, labels, 0)
        is_masked = labels != -100
        is_valid = batch.attention_mask[:num_samples] == 1

        log_strings = []

        for i in range(num_samples):
            valid_mask = is_valid[i]
            sentence = self._format_mlm_sequence(
                seq_ids=ids[i, valid_mask].tolist(),
                seq_preds=mlm_preds[i, valid_mask].tolist(),
                seq_trues=safe_labels[i, valid_mask].tolist(),
                seq_masks=is_masked[i, valid_mask].tolist(),
            )

            is_next_pred = nsp_preds[i].item() == 0
            is_next_true = batch.nsp_labels[i].item() == 0
            nsp_str = self._format_nsp_prediction(is_next_pred, is_next_true)

            log_strings.append(
                f"### Sample {i + 1}\n{sentence}\n\n**NSP:** {nsp_str}\n"
            )

        self.tracker.log_text(
            "Pretraining/Predictions", "\n---\n".join(log_strings), step=step
        )

    def _handle_checkpoint(self, step: int, current_loss: float):
        state = self._get_base_state_dict({"step": step, "loss": current_loss})
        self.save_checkpoint("last.pt", state)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_checkpoint("best.pt", state)
