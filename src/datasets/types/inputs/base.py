from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class BaseData(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Any:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        pass
