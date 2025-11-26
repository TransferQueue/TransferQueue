from abc import ABC, abstractmethod

from torch import Tensor


class TransferQueueStorageKVClient(ABC):
    """
    Abstract base class for storage client.
    Subclasses must implement the core methods: put, get, and clear.
    """

    @abstractmethod
    def put(self, keys: list[str], values: list[Tensor]) -> None:
        raise NotImplementedError("Subclasses must implement put")

    @abstractmethod
    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Tensor]:
        raise NotImplementedError("Subclasses must implement get")

    @abstractmethod
    def clear(self, keys: list[str]) -> None:
        raise NotImplementedError("Subclasses must implement clear")
