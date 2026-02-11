import pytest

from token_encoders.base import BaseTokenizer


@pytest.mark.parametrize(
    "tokenizer_fixture_name",
    [
        "trained_byte_pair_tokenizer",
        "trained_word_piece_tokenizer",
        "trained_rust_bpe_tokenizer",
        "trained_rust_word_piece_tokenizer",
    ],
)
@pytest.mark.parametrize(
    "special_token, property_name",
    [
        ("[PAD]", "pad_token_id"),
        ("[MASK]", "mask_token_id"),
        ("[CLS]", "cls_token_id"),
        ("[SEP]", "sep_token_id"),
        ("[UNK]", "unk_token_id"),
    ],
)
def test_special_tokens_encoding_all_tokenizers(
    tokenizer_fixture_name: str,
    special_token: str,
    property_name: str,
    request: pytest.FixtureRequest,
):
    tokenizer: BaseTokenizer = request.getfixturevalue(tokenizer_fixture_name)
    text = f"the {special_token} store"
    encoded = tokenizer.encode(text)
    expected_id = getattr(tokenizer, property_name)

    assert (
        expected_id in encoded
    ), f"{special_token} ID not found in encoded output for {tokenizer_fixture_name}"
    assert (
        encoded.count(expected_id) == 1
    ), f"{special_token} was split or duplicated in {tokenizer_fixture_name}"

    text_2 = f"[CLS]{special_token}[SEP]"
    encoded_2 = tokenizer.encode(text_2)
    assert encoded_2[0] == tokenizer.cls_token_id
    assert expected_id in encoded_2
    assert encoded_2[-1] == tokenizer.sep_token_id
