from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TokenizerSettings(BaseSettings):
    vocab_size: int = 15000
    special_tokens: list[str] = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]


class Settings(BaseSettings):
    tokenizer: TokenizerSettings = Field(default_factory=TokenizerSettings)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


SETTINGS = Settings()
