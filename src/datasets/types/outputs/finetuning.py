from dataclasses import dataclass

import torch

from datasets.types.outputs.base import BaseOutput


@dataclass
class FinetuningOutput(BaseOutput):
    labels: torch.Tensor
