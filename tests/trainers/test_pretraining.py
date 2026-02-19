from unittest.mock import MagicMock

import pytest

from data.base import BaseDataset
from modules.trainable import TrainableModel
from settings import PreTrainingSettings
from trainers.pretraining import PreTrainer


@pytest.fixture
def pretrainer(
    trainable_model: TrainableModel,
    mock_dataset: BaseDataset,
    pretraining_settings: PreTrainingSettings,
    mock_tracker: MagicMock,
) -> PreTrainer:
    return PreTrainer(
        pretraining_settings, trainable_model, mock_tracker, mock_dataset  # type: ignore
    )


def test_pretraining_runs_steps(
    pretrainer: PreTrainer,
    pretraining_settings: PreTrainingSettings,
    mock_tracker: MagicMock,
):
    pretrainer.train()
    assert mock_tracker.update_progress.call_count == pretrainer.total_steps
    exp_calls = pretrainer.total_steps / pretraining_settings.log_interval_steps
    assert mock_tracker.log_metric.call_count == 2 * exp_calls  # loss and LR logged
    save_calls = mock_tracker.save_artifact.call_args_list
    exp_calls = pretrainer.total_steps / pretraining_settings.save_interval_steps
    assert len(save_calls) == exp_calls + 1  # 2 last plus 1 best (same loss always)
    mock_tracker.close.assert_called_once()


def test_infinite_dataloader(pretrainer: PreTrainer, mock_dataset):
    iterator = pretrainer._get_infinite_dataloader()
    items = [next(iterator) for _ in range(5 * len(mock_dataset))]
    assert len(items) == 5 * len(mock_dataset)
