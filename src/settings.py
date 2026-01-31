from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TokenizerSettings(BaseSettings):
    vocab_size: int = 30522
    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    unk_token: str = "[UNK]"
    unused_tokens: int = 100

    @computed_field
    @property
    def special_tokens(self) -> list[str]:
        return [
            self.pad_token,
            self.mask_token,
            self.cls_token,
            self.sep_token,
            self.unk_token,
        ]


class BertSettings(BaseSettings):
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"


class LoaderSettings(BaseSettings):
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    max_seq_len: int = 512


class Settings(BaseSettings):
    bert: BertSettings = Field(default_factory=BertSettings)
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    loader: LoaderSettings = Field(default_factory=LoaderSettings)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @model_validator(mode="after")
    def check_sequence_length(self):
        phys_limit = self.bert.max_position_embeddings
        data_limit = self.loader.max_seq_len
        if data_limit > phys_limit:
            raise ValueError(
                f"Data loader max_seq_len ({data_limit}) cannot be larger than "
                f"Model max_position_embeddings ({phys_limit})."
            )
        return self


SETTINGS = Settings()
