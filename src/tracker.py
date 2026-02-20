import json
import sys
from datetime import datetime
from typing import Any

import torch
from pydantic_settings import BaseSettings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from settings import TrackerSettings


class ExperimentTracker:
    def __init__(self, settings: TrackerSettings, exp_settings: list[BaseSettings]):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.run_dir = settings.runs_dir / f"{settings.run_name}_{timestamp}"
        self.log_dir = self.run_dir / "tensorboard"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_exp_settings(exp_settings)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.progress_bar: tqdm | None = None
        self.global_step = 0

    def save_exp_settings(self, settings: list[BaseSettings]):
        sorted_settings = sorted(settings, key=lambda s: s.__class__.__name__)
        settings_dict = {s.__class__.__name__: s.model_dump() for s in sorted_settings}
        settings_path = self.run_dir / "settings.json"
        with settings_path.open("w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=2)

    def save_artifact(self, name: str, state_dict: dict):
        save_path = self.ckpt_dir / name
        torch.save(state_dict, save_path)

    def start_progress(self, total_steps: int, desc: str = "Training"):
        if self.progress_bar is not None:
            self.progress_bar.close()
        self.progress_bar = tqdm(
            total=total_steps, desc=desc, file=sys.stdout, dynamic_ncols=True
        )

    def update_progress(
        self, step_increment: int = 1, postfix: dict[str, Any] | None = None
    ):
        self.global_step += step_increment
        if self.progress_bar:
            self.progress_bar.update(step_increment)
            if postfix:
                formatted = {
                    k: f"{v:.4e}" if isinstance(v, float) else v
                    for k, v in postfix.items()
                }
                self.progress_bar.set_postfix(formatted)

    def log_metric(self, name: str, value: float, step: int | None = None):
        actual_step = step if step is not None else self.global_step
        self.writer.add_scalar(name, value, actual_step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_text(self, tag: str, text_string: str, step: int | None = None):
        actual_step = step if step is not None else self.global_step
        self.writer.add_text(tag, text_string, actual_step)

    def close(self):
        if self.progress_bar:
            self.progress_bar.close()
        self.writer.close()
