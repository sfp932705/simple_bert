from dataclasses import dataclass

import torch


@dataclass
class BaseOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
