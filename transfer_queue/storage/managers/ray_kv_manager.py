import ray
from typing import Any

from transfer_queue.storage.managers.base import KVStorageManager


class RayKVStorageManager(KVStorageManager):
    def __init__(self, config: dict[str, Any]):
        device_id = config.get("device_id", None)
        if device_id is None or not isinstance(device_id, int):
            raise ValueError("Missing or invalid 'device_id' in config")
        
        if not ray.is_initialized():
            ray.init()
        super().__init__(config)
