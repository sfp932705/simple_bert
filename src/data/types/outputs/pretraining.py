from dataclasses import dataclass

import torch

from data.types.outputs.base import BaseOutput


@dataclass
class PretrainingOutput(BaseOutput):
    mlm_labels: torch.Tensor
    nsp_labels: torch.Tensor
