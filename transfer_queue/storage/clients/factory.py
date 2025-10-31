# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Type
class TransferQueueStorageClient(ABC):
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


class StorageClientFactory:
    """
    Factory class for creating storage client instances.
    Uses a decorator-based registration mechanism to map client names to classes.
    """

    # Class variable: maps client names to their corresponding classes
    _registry: dict[str, TransferQueueStorageClient] = {}

    @classmethod
    def register(cls, client_type: str):
        """
        Decorator to register a concrete client class with the factory.
        Args:
            client_type (str): The name used to identify the client
        Returns:
            Callable: The decorator function that returns the original class
        """
        def decorator(client_class: TransferQueueStorageClient) -> TransferQueueStorageClient:
            cls._registry[client_type] = client_class
            return client_class
        return decorator

    @classmethod
    def create(cls, client_type: str, config: dict) -> TransferQueueStorageClient:
        """
        Create and return an instance of the storage client by name.
        Args:
            client_type (str): The registered name of the client
        Returns:
            StorageClientFactory: An instance of the requested client
        Raises:
            ValueError: If no client is registered with the given name
        """
        if client_type not in cls._registry:
            raise ValueError(f"Unknown StorageClient: {client_type}")
        return cls._registry[client_type](config)

# TODO: Dynamically register the storage client class based on the configuration
# Register storage clients
try:
    from .yuanrong_client import YRStorageClient
except ImportError:
    pass