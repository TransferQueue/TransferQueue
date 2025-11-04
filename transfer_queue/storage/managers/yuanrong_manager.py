from typing import Any

from transfer_queue.storage.managers.base import KVStorageManager


class YuanrongStorageManager(KVStorageManager):
    def __init__(self, config: dict[str, Any]):
        host = config.get("host", None)
        port = config.get("port", None)
        device_id = config.get("device_id", None)
        if host is None or not isinstance(host, str):
            raise ValueError("Missing or invalid 'host' in config")
        if port is None or not isinstance(port, int):
            raise ValueError("Missing or invalid 'port' in config")
        # TODO: device_id may be a list[int]
        if device_id is None or not isinstance(device_id, int):
            raise ValueError("Missing or invalid 'device_id' in config")
        super().__init__(config)
