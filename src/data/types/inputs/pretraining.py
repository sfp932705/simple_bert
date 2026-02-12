from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Self

from data.types.inputs.base import BaseData


@dataclass
class PretrainingCorpusData(BaseData):
    documents: list[list[str]]

    @classmethod
    def from_file(
        cls, path: Path, doc_separator: str = "<|||ITEM-SEPARATOR|||>"
    ) -> Self:
        content = path.read_text(encoding="utf-8")
        raw_documents = content.split(doc_separator)

        documents = []
        for raw_doc in raw_documents:
            parragraphs = [line.strip() for line in raw_doc.split("\n") if line.strip()]
            if parragraphs:
                documents.append(parragraphs)
        return cls(documents=documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx) -> list[str]:
        return self.documents[idx]

    def __iter__(self) -> Iterator[list[str]]:
        return iter(self.documents)
