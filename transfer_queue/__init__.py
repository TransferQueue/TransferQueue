import os

from .client import (
    AsyncTransferQueueClient,
    TransferQueueClient,
    process_zmq_server_info,
)
from .controller import TransferQueueController
from .metadata import BatchMeta
from .storage import TransferQueueStorageSimpleUnit

__all__ = [
    "AsyncTransferQueueClient",
    "BatchMeta",
    "TransferQueueClient",
    "TransferQueueController",
    "TransferQueueStorageSimpleUnit",
    "process_zmq_server_info",
]

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()
