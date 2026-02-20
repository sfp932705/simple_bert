from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from data.base import BaseDataset
from data.types.outputs.base import BaseOutput
from modules.bert.finetuning import FinetuningForwardPassOutput
from modules.bert.pretraining import PretrainingForwardPassOutput
from modules.trainable import TrainableModel
from settings import FinetuningSettings, PreTrainingSettings, TrainingSettings
from tracker import ExperimentTracker


@dataclass
class MockedOutput(BaseOutput):
    mlm_labels: torch.Tensor
    nsp_labels: torch.Tensor
    labels: torch.Tensor


@dataclass
class MockedForwardPassOutput(
    FinetuningForwardPassOutput, PretrainingForwardPassOutput
): ...


class MockModel(TrainableModel):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(5, 2)
        self.loss_tensor = torch.randn((1, 1))
        self.mlm_preds = torch.randn((3, 5, 2))
        self.nsp_preds = torch.randn((3, 5))
        self.logits = torch.randn((3, 5, 2))

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

    def train_forward_from_dataset_batch(
        self, batch: BaseOutput
    ) -> MockedForwardPassOutput:
        assert isinstance(batch, MockedOutput)
        loss = self.compute_loss_from_batch(batch)
        return MockedForwardPassOutput(
            loss=loss,
            mlm_preds=self.mlm_preds,
            nsp_preds=self.nsp_preds,
            logits=self.logits,
        )


class MockTokenizer:
    def __init__(self):
        # Using '_' as a mock delimiter to test the replace logic
        self.inverse_vocab = {
            0: "[PAD]",
            1: "[CLS]",
            2: "[SEP]",
            3: "[MASK]",
            4: "_the",
            5: "_dog",
            6: "_barked",
        }
        self.unk_token = "[UNK]"
        self.delimiter = "_"


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


class MockedDataset(BaseDataset):
    def __init__(self, batch: MockedOutput):
        self._batches = [batch, batch]
        self.tokenizer = None

    def loader(self) -> list[MockedOutput]:  # type: ignore
        return self._batches

    def __len__(self):
        return len(self._batches)


@pytest.fixture
def mock_tracker() -> MagicMock:
    return MagicMock(spec=ExperimentTracker)


@pytest.fixture
def trainable_model() -> MockModel:
    return MockModel()


@pytest.fixture
def mock_batch() -> MockedOutput:
    return MockedOutput(
        input_ids=torch.tensor([[1, 4, 3, 2, 0], [1, 5, 6, 2, 0]], dtype=torch.float),
        attention_mask=torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.float
        ),
        token_type_ids=torch.zeros(2, 5, dtype=torch.float),
        labels=torch.zeros(2, dtype=torch.float),
        mlm_labels=torch.tensor(
            [[-100, -100, 5, -100, -100], [-100, -100, -100, -100, -100]],
            dtype=torch.float,
        ),
        nsp_labels=torch.tensor([0, 1], dtype=torch.float),
    )


@pytest.fixture
def mock_dataset(
    mock_batch: MockedOutput, mock_tokenizer: MockTokenizer
) -> MockedDataset:
    dataset = MockedDataset(mock_batch)
    dataset.tokenizer = mock_tokenizer
    return dataset


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
