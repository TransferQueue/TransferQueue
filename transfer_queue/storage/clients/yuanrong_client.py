from transfer_queue.storage.clients.factory import StorageClientFactory, TransferQueueStorageClient
from typing import Any
from torch import Tensor
import torch
import datasystem
import torch_npu

# TODO: DSTensorClient.dev_mget has wrong
@StorageClientFactory.register("Yuanrong")
class YRStorageClient(TransferQueueStorageClient):
    """
    Storage client for YuanRong DataSystem.
    Communicates with the remote tensor storage service via DsTensorClient.
    All tensors must reside on NPU device.
    """
    def __init__(self, config: dict[str, Any]):
        self.host = config.get("host")
        self.port = config.get("port")
        self.device_id = config.get("device_id")
        self._ds_client = datasystem.DsTensorClient(self.host, self.port, self.device_id)
        self._ds_client.init()

    def _create_empty_tensorlist(self, shapes, dtypes):
        """
        Create a list of empty NPU tensors with given shapes and dtypes.
        Args:
            shapes (list): List of tensor shapes (e.g., [(3,), (2, 4)])
            dtypes (list): List of torch dtypes (e.g., [torch.float32, torch.int64])
        Returns:
            list: List of uninitialized NPU tensors
        """
        if len(dtypes) != len(shapes):
            raise ValueError('Length of dtypes must equal length of shapes')

        tensors: list[Tensor] = []
        for dtype, shape in zip(dtypes, shapes):
            tensor = torch.empty(shape, dtype=dtype).to(f'npu:{self.device_id}')
            tensors.append(tensor)
        return tensors

    def put(self, keys: list[str], values: list[Tensor]):
        """
        Store tensors to remote storage.
        Args:
            keys (list): List of string keys
            values (list): List of torch.Tensor on NPU
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        # TODO: Support the situation when the number of keys is greater than 10000
        if len(keys) > 10000:
            raise NotImplementedError('We will support the number of keys greater than 10000 int the future')

        for value in values:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(value)}")
            if value.device.type != 'npu':
                raise ValueError(f"Tensor is on {value.device}, not on NPU")

        self._ds_client.dev_mset(keys, values)

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Tensor]:
        """
        Retrieve tensors from remote storage.
        Args:
            keys (list): List of keys to fetch
            shapes (list): Expected shapes of returned tensors
            dtypes (list): Expected dtypes of returned tensors
        Returns:
            list: List of retrieved NPU tensors
        """
        if shapes is None:
            raise ValueError('Yuanrong storage client needs Expected shapes of returned tensors')
        if dtypes is None:
            raise ValueError('Yuanrong storage client needs Expected dtypes of returned tensors')
        if len(dtypes) != len(shapes):
            raise ValueError('Length of dtypes must equal length of shapes')

        values: list[Tensor] = self._create_empty_tensorlist(shapes=shapes, dtypes=dtypes)

        # TODO: Support the situation when the number of keys is greater than 10000
        if len(keys) > 10000:
            raise NotImplementedError('We will support the number of keys greater than 10000 int the future')

        # Timeout set to 2000ms
        self._ds_client.dev_mget(keys, values, 2000)
        return values

    def clear(self, keys: list[str]):
        """
        Delete entries from storage by keys.
        Args:
            keys (list): List of keys to delete
        """
        self._ds_client.dev_delete(keys)