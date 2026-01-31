import torch
from datasets.inference import InferenceDataset
from datasets.types.inputs.inference import InferenceData

def test_inference_dataset(tokenizer, settings):
    # 1. Setup Data
    data = InferenceData(texts=["test query"])
    
    # 2. Init Dataset
    ds = InferenceDataset(data, tokenizer, settings)
    
    # 3. Validation
    item = ds[0]
    
    # Ensure no labels are present
    assert not hasattr(item, "labels")
    
    # Check Tensors
    assert isinstance(item.input_ids, torch.Tensor)
    assert isinstance(item.attention_mask, torch.Tensor)
    
    # Check Shapes
    assert item.input_ids.shape[0] == settings.max_seq_len
    assert item.token_type_ids.shape[0] == settings.max_seq_len
