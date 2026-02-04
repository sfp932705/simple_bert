from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from datasets.types.outputs.base import BaseOutput
from modules.trainable import TrainableModel
from settings import TrainerSettings
from trainer import Trainer


@dataclass
class MockedOutput(BaseOutput):
    pass


class MockModel(TrainableModel):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def compute_loss_from_batch(self, batch: MockedOutput) -> torch.Tensor:
        return batch.input_ids.sum().float().requires_grad_(True)

    def compute_loss_from_dataset_batch(self, batch: BaseOutput) -> torch.Tensor:
        assert isinstance(batch, MockedOutput)
        return self.compute_loss_from_batch(batch)


@pytest.fixture
def trainer_settings(tmp_path: Path) -> TrainerSettings:
    return TrainerSettings(
        num_train_epochs=2,
        checkpoint_dir=tmp_path / "checkpoints",
        log_interval=1,
        device="cpu",
        learning_rate=1e-3,
    )


@pytest.fixture
def mock_loader() -> list[MockedOutput]:
    dummy_batch = MockedOutput(
        input_ids=torch.randn(4, 10),
        attention_mask=torch.ones(4, 10),
        token_type_ids=torch.zeros(4, 10),
    )
    return [dummy_batch]


@pytest.fixture
def trainable_model() -> MockModel:
    return MockModel()


def test_trainer_initialization(
    trainable_model: MockModel,
    mock_loader: list[MockedOutput],
    trainer_settings: TrainerSettings,
) -> None:
    trainer = Trainer(trainable_model, mock_loader, trainer_settings)  # type: ignore
    assert trainer.model == trainable_model
    assert trainer.optimizer is not None
    assert trainer.device == torch.device("cpu")


def test_trainer_runs_epochs(
    trainable_model: MockModel,
    mock_loader: list[MockedOutput],
    trainer_settings: TrainerSettings,
) -> None:
    optimizer = torch.optim.SGD(trainable_model.parameters(), lr=0.1)
    optimizer.step = MagicMock(wraps=optimizer.step)  # type: ignore
    trainer = Trainer(
        trainable_model, mock_loader, trainer_settings, optimizer=optimizer  # type: ignore
    )
    trainer.train()
    assert optimizer.step.called  # type: ignore
    expected_calls = len(mock_loader) * trainer_settings.num_train_epochs
    assert optimizer.step.call_count == expected_calls  # type: ignore
    expected_ckpt = trainer_settings.checkpoint_dir / "checkpoint_epoch_0.pt"
    assert expected_ckpt.exists()


def test_trainer_moves_batch_to_device(
    trainable_model: MockModel,
    mock_loader: list[MockedOutput],
    trainer_settings: TrainerSettings,
) -> None:
    mock_batch = mock_loader[0]
    mock_batch.to = MagicMock(return_value=mock_batch)  # type: ignore
    trainer_settings.num_train_epochs = 1
    trainer = Trainer(trainable_model, [mock_batch], trainer_settings)  # type: ignore
    trainer.train()
    mock_batch.to.assert_called_with(torch.device("cpu"))  # type: ignore


def test_trainer_saves_state_dict(
    trainable_model: MockModel,
    mock_loader: list[MockedOutput],
    trainer_settings: TrainerSettings,
) -> None:
    trainer = Trainer(trainable_model, mock_loader, trainer_settings)  # type: ignore
    trainer.train()
    ckpt_path = trainer_settings.checkpoint_dir / "checkpoint_epoch_0.pt"
    checkpoint = torch.load(ckpt_path)

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert checkpoint["epoch"] == 0
