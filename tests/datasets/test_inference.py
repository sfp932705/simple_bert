import torch
from data.inference import InferenceDataset
from data.types.inputs.inference import InferenceData
from settings import LoaderSettings
from token_encoders.bpe import BPETokenizer


def test_inference_dataset(tokenizer: BPETokenizer, settings: LoaderSettings):
    data = InferenceData(texts=["test query"])
    ds = InferenceDataset(data, tokenizer, settings)
    item = ds[0]
    assert not hasattr(item, "labels")
    assert isinstance(item.input_ids, torch.Tensor)
    assert isinstance(item.attention_mask, torch.Tensor)
    assert item.input_ids.shape[0] == settings.max_seq_len
    assert item.token_type_ids.shape[0] == settings.max_seq_len
