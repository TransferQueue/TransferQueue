import logging
import os
import pickle
from typing import Any

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

DS_CLIENT_KEYS_LIMIT: int = 1999
YUANRONG_DATASYSTEM_IMPORTED: bool = True
TORCH_NPU_IMPORTED: bool = True
try:
    import datasystem
except ImportError:
    YUANRONG_DATASYSTEM_IMPORTED = False
try:
    import torch_npu  # noqa: F401
except ImportError:
    TORCH_NPU_IMPORTED = False


@StorageClientFactory.register("YuanrongStorageClient")
class YuanrongRStorageClient(TransferQueueStorageKVClient):
    """
    Storage client for YuanRong DataSystem.

    Supports storing both:
    - NPU tensors via DsTensorClient (for high performance).
    - General objects (CPU tensors, str, bool, list, etc.) via KVClient with pickle serialization.
    """

    def __init__(self, config: dict[str, Any]):
        if not YUANRONG_DATASYSTEM_IMPORTED:
            raise ImportError("YuanRong DataSystem not installed.")

        self.host = config.get("host")
        self.port = config.get("port")

        self.device_id = None
        self._npu_ds_client = None
        self._cpu_ds_client = None

        if not TORCH_NPU_IMPORTED:
            logger.warning(
                "'torch_npu' import failed. "
                "This results in the inability to quickly store and retrieve tensors on the NPU side,"
                "which may affect performance."
            )
        elif not torch.npu.is_available():
            logger.warning(
                "NPU is not available. "
                "This results in the inability to quickly store and retrieve tensors on the NPU side, "
                "which may affect performance."
            )
        else:
            self.device_id = torch.npu.current_device()
            self._npu_ds_client = datasystem.DsTensorClient(self.host, self.port, self.device_id)
            self._npu_ds_client.init()

        self._cpu_ds_client = datasystem.KVClient(self.host, self.port)
        self._cpu_ds_client.init()

    def npu_ds_client_is_available(self):
        return self._npu_ds_client is not None

    def cpu_ds_client_is_available(self):
        return self._cpu_ds_client is not None

    def _create_empty_npu_tensorlist(self, shapes, dtypes):
        """
        Create a list of empty NPU tensors with given shapes and dtypes.

        Args:
            shapes (list): List of tensor shapes (e.g., [(3,), (2, 4)])
            dtypes (list): List of torch dtypes (e.g., [torch.float32, torch.int64])
        Returns:
            list: List of uninitialized NPU tensors
        """
        tensors: list[Tensor] = []
        for shape, dtype in zip(shapes, dtypes, strict=False):
            tensor = torch.empty(shape, dtype=dtype, device=f"npu:{self.device_id}")
            tensors.append(tensor)
        return tensors

    def _batch_put(self, keys: list[str], values: list[Any]):
        if self.npu_ds_client_is_available():
            cpu_keys = []
            cpu_values = []
            npu_keys = []
            npu_values = []
            for key, value in zip(keys, values, strict=False):
                if isinstance(value, torch.Tensor) and value.device.type == "npu":
                    if not value.is_contiguous():
                        raise ValueError(f"NPU Tensor is not contiguous: {value}")
                    npu_keys.append(key)
                    npu_values.append(value)
                else:
                    cpu_keys.append(key)
                    cpu_values.append(pickle.dumps(value))

            if npu_keys:
                # _npu_ds_client.dev_mset doesn't support to overwrite
                try:
                    self._npu_ds_client.dev_delete(npu_keys)
                except Exception as e:
                    logger.warning(f"dev_delete error({e}) before dev_mset")

                self._npu_ds_client.dev_mset(npu_keys, npu_values)

            if cpu_keys:
                self._cpu_ds_client.mset(cpu_keys, cpu_values)
        else:
            values = [pickle.dumps(value) for value in values]
            self._cpu_ds_client.mset(keys, values)

    def put(self, keys: list[str], values: list[Any]):
        """
        Store data(npu tensors, cpu tensors, nontensor, python basic objects) to remote storage.
        Args:
            keys (list): List of string keys
            values (list): List of torch.Tensor on NPU
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        # Each time, process at most DS_CLIENT_KEYS_LIMIT keys, and this is done for a total of count times.
        # The calculation below uses ceiling division (i.e., integer division rounded up).
        total_count = (len(keys) + DS_CLIENT_KEYS_LIMIT - 1) // DS_CLIENT_KEYS_LIMIT
        for i in range(total_count):
            start_idx = DS_CLIENT_KEYS_LIMIT * i
            end_idx = min(DS_CLIENT_KEYS_LIMIT * (i + 1), len(keys))
            self._batch_put(keys[start_idx:end_idx], values[start_idx:end_idx])

    def _batch_get(self, keys, shapes, dtypes) -> list[Any]:
        if self.npu_ds_client_is_available():
            cpu_keys = []
            npu_keys = []
            npu_dtypes = []
            npu_shapes = []
            for shape, dtype, key in zip(shapes, dtypes, keys):
                if dtype is not None:
                    npu_shapes.append(shape)
                    npu_dtypes.append(dtype)
                    npu_keys.append(key)
                else:
                    cpu_keys.append(key)

            # Note: _npu_ds_client.dev_mget and _cpu_ds_client.get(keys) is assumed to return values in the same order as keys
            failed_keys = []
            npu_values = []

            if npu_keys:
                npu_values = self._create_empty_npu_tensorlist(npu_shapes, npu_dtypes)
                try:
                    failed_keys = self._npu_ds_client.dev_mget(npu_keys, npu_values)
                    failed_keys = [f_key.rsplit(',', 1)[0] for f_key in failed_keys]
                except Exception:
                    failed_keys = npu_keys
                    npu_keys = []

            if failed_keys:
                cpu_keys.extend(failed_keys)
            cpu_values = []
            if cpu_keys:
                cpu_values = self._cpu_ds_client.get(cpu_keys)

            key_to_position = {key: position for position, key in enumerate(keys)}

            values = [None] * len(keys)

            # npu_keys have failed_keys
            for key, value in zip(npu_keys, npu_values, strict=False):
                values[key_to_position[key]] = value
            for key, value in zip(cpu_keys, cpu_values, strict=False):
                values[key_to_position[key]] = pickle.loads(value)
        else:
            values = self._cpu_ds_client.get(keys)
            values = [pickle.loads(value) for value in values]

        return values

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Any]:
        """
        Retrieve data from remote storage.
        Args:
            keys (list): List of keys to fetch
            shapes (list): Expected shapes of returned data
            dtypes (list): Expected dtypes of returned data
        Returns:
            list: List of retrieved data
        """
        if shapes is None:
            raise ValueError("Yuanrong storage client needs Expected shapes of returned data")
        if dtypes is None:
            raise ValueError("Yuanrong storage client needs Expected dtypes of returned data")
        if len(dtypes) != len(shapes) or len(keys) != len(shapes):
            raise ValueError("Length of dtypes must equal length of shapes")

        # Each time, process at most DS_CLIENT_KEYS_LIMIT keys, and this is done for a total of count times.
        # The calculation below uses ceiling division (i.e., integer division rounded up).
        total_count = (len(keys) + DS_CLIENT_KEYS_LIMIT - 1) // DS_CLIENT_KEYS_LIMIT
        values = []
        for i in range(total_count):
            start_idx = DS_CLIENT_KEYS_LIMIT * i
            end_idx = min(DS_CLIENT_KEYS_LIMIT * (i + 1), len(keys))
            values.extend(
                self._batch_get(keys[start_idx:end_idx], shapes[start_idx:end_idx], dtypes[start_idx:end_idx])
            )
        return values

    def _batch_clear(self, keys):
        if self.npu_ds_client_is_available():
            # Delete from NPU storage; get keys that failed to delete
            failed_keys = self._npu_ds_client.dev_delete(keys)
            # Attempt to delete the failed keys from CPU storage as a fallback
            if failed_keys:
                self._cpu_ds_client.delete(failed_keys)
        else:
            self._cpu_ds_client.delete(keys)

    def clear(self, keys: list[str]):
        """
        Delete entries from storage by keys.
        Args:
            keys (list): List of keys to delete
        """
        total_count = (len(keys) + DS_CLIENT_KEYS_LIMIT - 1) // DS_CLIENT_KEYS_LIMIT
        for i in range(total_count):
            start_idx = DS_CLIENT_KEYS_LIMIT * i
            end_idx = min(DS_CLIENT_KEYS_LIMIT * (i + 1), len(keys))
            self._batch_clear(keys[start_idx:end_idx])
