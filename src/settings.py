from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VocabSettings(BaseSettings):
    vocab_size: int = 30522


class LayerCommonSettings(VocabSettings):
    hidden_size: int = 768
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12


class AttentionSettings(LayerCommonSettings):
    num_attention_heads: int = 12
    attention_probs_dropout_prob: float = 0.1


class EmbeddingSettings(LayerCommonSettings):
    max_position_embeddings: int = 512
    type_vocab_size: int = 2


class TokenizerSettings(VocabSettings):
    vocab_size: int = 30522
    special_tokens: list[str] = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]
    unused_tokens: int = 100


class Settings(BaseSettings):
    attention: AttentionSettings = Field(default_factory=AttentionSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


SETTINGS = Settings()
