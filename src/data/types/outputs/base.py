from dataclasses import dataclass, replace
from typing import Self

import torch


@dataclass
class BaseOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    def to(self, device: torch.device | str) -> Self:
        changes = {}
        for field, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                changes[field] = value.to(device)
        return replace(self, **changes)
