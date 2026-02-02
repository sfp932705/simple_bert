import pytest
from pydantic import ValidationError

from settings import BertSettings, LoaderSettings, Settings


@pytest.fixture
def bert_config_small() -> BertSettings:
    return BertSettings(max_position_embeddings=128)


def test_settings_validation_error_raises(bert_config_small: BertSettings):
    loader_invalid = LoaderSettings(max_seq_len=256)
    with pytest.raises(ValidationError) as exc_info:
        Settings(bert=bert_config_small, loader=loader_invalid)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    msg = errors[0]["msg"]
    assert "Data loader max_seq_len (256)" in msg
    assert "Model max_position_embeddings (128)" in msg
