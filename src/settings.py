from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TokenizerSettings(BaseSettings):
    vocab_size: int = 30522
    special_tokens: list[str] = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]
    unused_tokens: int = 100


class Settings(BaseSettings):
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


SETTINGS = Settings()
