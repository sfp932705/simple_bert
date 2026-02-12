from dataclasses import dataclass

import torch

from data.types.outputs.base import BaseOutput


@dataclass
class FinetuningOutput(BaseOutput):
    labels: torch.Tensor
