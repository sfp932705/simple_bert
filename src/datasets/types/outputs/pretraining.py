from dataclasses import dataclass

import torch

from datasets.types.outputs.base import BaseOutput


@dataclass
class PretrainingOutput(BaseOutput):
    mlm_labels: torch.Tensor
    nsp_labels: torch.Tensor
