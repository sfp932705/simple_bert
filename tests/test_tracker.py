import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from settings import TrackerSettings
from tracker import ExperimentTracker


@pytest.fixture
def tracker_settings(tmp_path: Path) -> TrackerSettings:
    return TrackerSettings(run_name="integration_test", runs_dir=tmp_path / "runs")


@pytest.fixture
def tracker(tracker_settings: TrackerSettings) -> ExperimentTracker:
    return ExperimentTracker(tracker_settings)


def test_directory_structure_creation(tracker: ExperimentTracker):
    assert tracker.run_dir.exists()
    assert tracker.log_dir.exists()
    assert tracker.ckpt_dir.exists()
    assert tracker.run_dir.is_dir()
    assert tracker.log_dir.is_dir()


def test_logging_metrics_creates_event_file(tracker: ExperimentTracker):
    tracker.log_metric("loss", 0.5, step=1)
    tracker.log_metrics({"acc": 0.9, "lr": 1e-3}, step=1)
    tracker.close()
    event_files = list(tracker.log_dir.glob("events.out.tfevents*"))
    assert len(event_files) > 0, "No TensorBoard event file was created."
    assert event_files[0].stat().st_size > 0, "TensorBoard event file is empty."


def test_saving_artifact_writes_file(tracker: ExperimentTracker):
    state_dict = {"model": torch.tensor([1.0, 2.0, 3.0])}
    filename = "test_model.pt"
    tracker.save_artifact(filename, state_dict)
    expected_path = tracker.ckpt_dir / filename
    assert expected_path.exists()
    loaded_state = torch.load(expected_path)
    assert "model" in loaded_state
    assert torch.equal(loaded_state["model"], state_dict["model"])


def test_progress_bar_logic_does_not_crash(tracker: ExperimentTracker):
    tracker.start_progress(total_steps=5, desc="First Test Bar")
    with patch("sys.stdout"):
        tracker.start_progress(total_steps=5, desc="Second Test Bar")
        for i in range(5):
            tracker.update_progress(step_increment=1, postfix={"loss": 0.1})
        assert tracker.global_step == 5
        tracker.close()


def test_multiple_runs_create_unique_folders(tracker_settings: TrackerSettings):
    t1 = ExperimentTracker(tracker_settings)
    path1 = t1.run_dir
    time.sleep(1.1)
    t2 = ExperimentTracker(tracker_settings)
    path2 = t2.run_dir
    assert path1 != path2
    assert path1.exists()
    assert path2.exists()
