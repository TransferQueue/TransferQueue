import logging
import os
from typing import Any

from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


@TransferQueueStorageManagerFactory.register("YuanrongStorageManager")
class YuanrongStorageManager(KVStorageManager):
    def __init__(self, config: dict[str, Any]):
        host = config.get("host", None)
        port = config.get("port", None)
        client_name = config.get("client_name", None)

        if host is None or not isinstance(host, str):
            raise ValueError("Missing or invalid 'host' in config")
        if port is None or not isinstance(port, int):
            raise ValueError("Missing or invalid 'port' in config")
        if client_name is None:
            logger.info("Missing 'client_name' in config, using default value('YuanrongStorageClient')")
            config["client_name"] = "YuanrongStorageClient"
        elif client_name != "YuanrongStorageClient":
            raise ValueError(f"Invalid 'client_name': {client_name} in config. Expecting 'YuanrongStorageClient'")
        super().__init__(config)
