from copy import deepcopy
from unittest.mock import MagicMock

import pytest
import torch

from datasets.types.outputs.base import BaseOutput
from modules.trainable import TrainableModel
from settings import FinetuningSettings
from trainers.finetuning import FinetuningTrainer


@pytest.fixture
def finetuning_trainer(
    trainable_model: TrainableModel,
    mock_loader: list[BaseOutput],
    finetuning_settings: FinetuningSettings,
    mock_tracker: MagicMock,
) -> FinetuningTrainer:
    return FinetuningTrainer(
        finetuning_settings, trainable_model, mock_tracker, mock_loader  # type: ignore
    )


@pytest.fixture
def finetuning_trainer_with_mocked_optimizer(
    finetuning_trainer: FinetuningTrainer,
) -> FinetuningTrainer:
    params = finetuning_trainer.model.parameters()
    optimizer = torch.optim.SGD(params, lr=0.1)
    optimizer.step = MagicMock(wraps=optimizer.step)  # type: ignore
    finetuning_trainer.optimizer = optimizer
    return finetuning_trainer


@pytest.fixture
def finetuning_trainer_with_validation_loader(
    finetuning_trainer: FinetuningTrainer,
) -> FinetuningTrainer:
    finetuning_trainer.val_loader = deepcopy(finetuning_trainer.train_loader)
    return finetuning_trainer


def test_finetuning_initialization(
    finetuning_trainer: FinetuningTrainer,
    trainable_model: TrainableModel,
    finetuning_settings: FinetuningSettings,
    mock_loader: list[BaseOutput],
):
    assert finetuning_trainer.model == trainable_model
    assert finetuning_trainer.optimizer is not None
    assert finetuning_trainer.total_steps == finetuning_settings.num_train_epochs * len(
        mock_loader
    )


def test_finetuning_runs_epochs(
    finetuning_trainer_with_mocked_optimizer: FinetuningTrainer,
    finetuning_settings: FinetuningSettings,
    mock_tracker: MagicMock,
):
    epochs = finetuning_settings.num_train_epochs
    trainer = finetuning_trainer_with_mocked_optimizer
    trainer.train()
    assert trainer.optimizer.step.call_count == trainer.total_steps  # type: ignore
    assert mock_tracker.start_progress.call_count == epochs
    assert mock_tracker.update_progress.call_count == trainer.total_steps
    mock_tracker.close.assert_called_once()


def test_finetuning_saves_checkpoints_via_tracker(
    finetuning_trainer_with_validation_loader: FinetuningTrainer,
    trainable_model: TrainableModel,
    mock_loader: list[BaseOutput],
    finetuning_settings: FinetuningSettings,
    mock_tracker: MagicMock,
):
    trainer = finetuning_trainer_with_validation_loader
    trainer.train()
    save_calls = mock_tracker.save_artifact.call_args_list
    last_pt_saves = [call for call in save_calls if call[0][0] == "last.pt"]
    assert len(last_pt_saves) == finetuning_settings.num_train_epochs
    saved_state = last_pt_saves[0][0][1]
    assert "model_state_dict" in saved_state
    assert "epoch" in saved_state


def test_finetuning_evaluation_logic(
    finetuning_trainer_with_validation_loader: FinetuningTrainer,
):
    acc, loss = finetuning_trainer_with_validation_loader.evaluate()
    assert isinstance(acc, float)
    assert isinstance(loss, float)
