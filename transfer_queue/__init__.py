import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()

from .data_system import (
    AsyncTransferQueueClient,
    TransferQueueClient,
    TransferQueueController,
    TransferQueueStorageSimpleUnit,
    process_zmq_server_info,
)

__all__ = [
    "TransferQueueClient",
    "AsyncTransferQueueClient",
    "TransferQueueController",
    "TransferQueueStorageSimpleUnit",
    "process_zmq_server_info",
]
