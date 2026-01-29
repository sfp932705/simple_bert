from dataclasses import dataclass
from typing import Iterator, Self

from datasets.types.base import BaseData


@dataclass
class PretrainingCorpusData(BaseData):
    documents: list[list[str]]

    @classmethod
    def from_file(cls, path: str, sentence_separator: str = "|||") -> Self:
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(line.strip().split(sentence_separator))
        return cls(documents=documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx) -> list[str]:
        return self.documents[idx]

    def __iter__(self) -> Iterator[list[str]]:
        return iter(self.documents)
