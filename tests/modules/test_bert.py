import torch

from modules.bert import Bert


def test_bert_model_forward_pass(
    bert: Bert,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
):
    sequence_output, pooled_output = bert(sample_indices)
    assert sequence_output.shape == (batch_size, seq_len, hidden_size)
    assert pooled_output.shape == (batch_size, hidden_size)


def test_bert_model_masking_logic(
    bert: Bert,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
):
    mask = torch.ones((batch_size, seq_len))
    mask[:, -2:] = 0
    seq_out, pool_out = bert(sample_indices, attention_mask=mask)
    assert seq_out.shape == (batch_size, seq_len, hidden_size)


def test_bert_model_token_types(
            bert: Bert,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    hidden_size: int,


):
    bert.eval()
    types_A = torch.zeros((1, seq_len), dtype=torch.long)
    out_A, _ = bert(sample_indices, token_type_ids=types_A)
    types_B = torch.ones((1, seq_len), dtype=torch.long)
    out_B, _ = bert(sample_indices, token_type_ids=types_B)
    assert not torch.allclose(out_A, out_B)
