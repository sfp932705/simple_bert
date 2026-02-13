from dataclasses import dataclass
from typing import Iterator, Self

from data.types.inputs.base import BaseData


@dataclass
class InferenceData(BaseData):
    texts: list[str]

    @classmethod
    def from_file(cls, path: str) -> Self:
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return cls(texts=texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> str:
        return self.texts[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.texts)
