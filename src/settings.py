from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TokenizerSettings(BaseSettings):
    vocab_size: int = 30522
    special_tokens: list[str] = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]
    unused_tokens: int = 100


class AttentionSettings(BaseSettings):
    vocab_size: int = 30522
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12


class Settings(BaseSettings):
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    attention: AttentionSettings = Field(default_factory=AttentionSettings)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


SETTINGS = Settings()
