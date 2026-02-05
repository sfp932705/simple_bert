import torch
from torch.optim.lr_scheduler import LambdaLR


class BertScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self._scheduler = self._get_linear_schedule_with_warmup()

    def step(self):
        self._scheduler.step()

    def get_last_lr(self) -> list[float]:
        return self._scheduler.get_last_lr()

    def state_dict(self) -> dict:
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict):
        self._scheduler.load_state_dict(state_dict)

    def _get_linear_schedule_with_warmup(self) -> LambdaLR:
        def lr_lambda(current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            return max(
                0.0,
                float(self.num_training_steps - current_step)
                / float(max(1, self.num_training_steps - self.num_warmup_steps)),
            )

        return LambdaLR(self.optimizer, lr_lambda)
