from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from datasets.types.outputs.base import BaseOutput
from modules.trainable import TrainableModel
from settings import FinetuningSettings, PreTrainingSettings, TrainingSettings
from tracker import ExperimentTracker


@dataclass
class MockedOutput(BaseOutput):
    mlm_labels: torch.Tensor
    nsp_labels: torch.Tensor
    labels: torch.Tensor


class MockModel(TrainableModel):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)
        self.loss_tensor = torch.randn((1, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layer(input_ids)

    def compute_loss(
        self,
        prediction_scores: torch.Tensor | None = None,
        seq_relationship_score: torch.Tensor | None = None,
        mlm_labels: torch.Tensor | None = None,
        nsp_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_tensor

    def compute_loss_from_batch(self, batch: MockedOutput) -> torch.Tensor:
        return batch.input_ids.sum().float().requires_grad_(True)

    def compute_loss_from_dataset_batch(self, batch: BaseOutput) -> torch.Tensor:
        assert isinstance(batch, MockedOutput)
        return self.compute_loss_from_batch(batch)


@pytest.fixture
def mock_tracker() -> MagicMock:
    return MagicMock(spec=ExperimentTracker)


@pytest.fixture
def trainable_model() -> MockModel:
    return MockModel()


@pytest.fixture
def mock_batch() -> MockedOutput:
    return MockedOutput(
        input_ids=torch.randn(4, 10),
        attention_mask=torch.ones(4, 10),
        token_type_ids=torch.zeros(4, 10),
        labels=torch.zeros(4),
        mlm_labels=torch.zeros(4, 10),
        nsp_labels=torch.zeros(4, 10),
    )


@pytest.fixture
def mock_loader(mock_batch: MockedOutput) -> list[MockedOutput]:
    return [mock_batch, mock_batch]


@pytest.fixture
def training_settings() -> TrainingSettings:
    return TrainingSettings()


@pytest.fixture
def finetuning_settings() -> FinetuningSettings:
    return FinetuningSettings(num_train_epochs=2)


@pytest.fixture
def pretraining_settings() -> PreTrainingSettings:
    return PreTrainingSettings(
        total_steps=10, log_interval_steps=2, save_interval_steps=5
    )
