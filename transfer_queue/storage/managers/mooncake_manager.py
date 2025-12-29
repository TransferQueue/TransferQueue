import logging
import os
from typing import Any

from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


@TransferQueueStorageManagerFactory.register("MooncakeStorageManager")
class MooncakeStorageManager(KVStorageManager):
    def __init__(self, config: dict[str, Any]):
        metadata_server = config.get("metadata_server", None)
        master_server_address = config.get("master_server_address", None)
        client_name = config.get("client_name", None)

        if metadata_server is None or not isinstance(metadata_server, str):
            raise ValueError("Missing or invalid 'metadata_server' in config")
        if master_server_address is None or not isinstance(master_server_address, str):
            raise ValueError("Missing or invalid 'master_server_address' in config")
        if client_name is None:
            logger.info("Missing 'client_name' in config, using default value('MooncakeStorageClient')")
            config["client_name"] = "MooncakeStorageClient"
        elif client_name != "MooncakeStorageClient":
            raise ValueError(f"Invalid 'client_name': {client_name} in config. Expecting 'MooncakeStorageClient'")
        super().__init__(config)
