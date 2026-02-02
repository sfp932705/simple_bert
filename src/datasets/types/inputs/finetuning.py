from dataclasses import dataclass
from typing import Iterator, Self

from datasets.types.inputs.base import BaseData


@dataclass
class LabeledTextSample:
    text: str
    label: int


@dataclass
class FinetuningData(BaseData):
    samples: list[LabeledTextSample]

    @classmethod
    def from_lists(cls, texts: list[str], labels: list[int]) -> Self:
        if len(texts) != len(labels):
            raise ValueError("Texts and Labels must have the same length.")
        return cls(samples=[LabeledTextSample(t, lbl) for t, lbl in zip(texts, labels)])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> LabeledTextSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[LabeledTextSample]:
        return iter(self.samples)
