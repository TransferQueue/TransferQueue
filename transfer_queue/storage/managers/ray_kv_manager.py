import ray
from typing import Any

from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory

@TransferQueueStorageManagerFactory.register("RAY")
class RayKVStorageManager(KVStorageManager):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
