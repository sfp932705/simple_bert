from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TokenizerSettings(BaseSettings):
    vocab_size: int = 30522
    special_tokens: list[str] = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]
    unused_tokens: int = 100


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


class Settings(BaseSettings):
    bert: BertSettings = Field(default_factory=BertSettings)
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


SETTINGS = Settings()
