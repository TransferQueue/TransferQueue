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

MOONCAKE_STORE_IMPORTED: bool = True
try:
    from mooncake.store import MooncakeDistributedStore
except ImportError:
    MOONCAKE_STORE_IMPORTED = False

BATCH_SIZE_LIMIT: int = 500


@StorageClientFactory.register("MooncakeStorageClient")
class MooncakeStorageClient(TransferQueueStorageKVClient):
    def __init__(self, config: dict[str, Any]):
        if not MOONCAKE_STORE_IMPORTED:
            raise ImportError(
                "Mooncake Store not installed. "
                "Please install via: pip install mooncake-transfer-engine"
            )

        self.local_hostname = config.get("local_hostname", "localhost")
        self.metadata_server = config.get("metadata_server")
        self.global_segment_size = config.get("global_segment_size", 512 * 1024 * 1024)
        self.local_buffer_size = config.get("local_buffer_size", 128 * 1024 * 1024)
        self.protocol = config.get("protocol", "tcp")
        self.device_name = config.get("device_name", "")
        self.master_server_address = config.get("master_server_address")

        if self.metadata_server is None:
            raise ValueError("Missing 'metadata_server' in config")
        if self.master_server_address is None:
            raise ValueError("Missing 'master_server_address' in config")

        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            self.local_hostname,
            self.metadata_server,
            self.global_segment_size,
            self.local_buffer_size,
            self.protocol,
            self.device_name,
            self.master_server_address,
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake store setup failed with error code: {ret}")

    def put(self, keys: list[str], values: list[Any]):
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        tensor_keys = []
        tensor_values = []
        non_tensor_keys = []
        non_tensor_values = []

        for key, value in zip(keys, values, strict=True):
            if isinstance(value, torch.Tensor):
                tensor = value.contiguous()
                tensor_keys.append(key)
                tensor_values.append(tensor)
            else:
                non_tensor_keys.append(key)
                non_tensor_values.append(pickle.dumps(value))

        if tensor_keys:
            self._batch_put_tensors(tensor_keys, tensor_values)

        if non_tensor_keys:
            self._batch_put_bytes(non_tensor_keys, non_tensor_values)

    def _batch_put_tensors(self, keys: list[str], tensors: list[Tensor]):
        values_bytes = []
        for tensor in tensors:
            t = tensor.detach().cpu()
            if t.dtype == torch.bfloat16:
                t = t.view(torch.int16)
            tensor_bytes = t.numpy().tobytes()
            values_bytes.append(tensor_bytes)

        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i:i + BATCH_SIZE_LIMIT]
            batch_values = values_bytes[i:i + BATCH_SIZE_LIMIT]
            ret = self._store.put_batch(batch_keys, batch_values)
            if ret != 0:
                raise RuntimeError(f"put_batch failed with error code: {ret}")

    def _batch_put_bytes(self, keys: list[str], values: list[bytes]):
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i:i + BATCH_SIZE_LIMIT]
            batch_values = values[i:i + BATCH_SIZE_LIMIT]
            ret = self._store.put_batch(batch_keys, batch_values)
            if ret != 0:
                raise RuntimeError(f"put_batch failed with error code: {ret}")

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Any]:
        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStorageClient needs shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        tensor_indices = []
        non_tensor_indices = []

        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)

        results = [None] * len(keys)

        if tensor_indices:
            tensor_keys = [keys[i] for i in tensor_indices]
            tensor_shapes = [shapes[i] for i in tensor_indices]
            tensor_dtypes = [dtypes[i] for i in tensor_indices]
            tensor_results = self._batch_get_tensors(tensor_keys, tensor_shapes, tensor_dtypes)
            for idx, tensor in zip(tensor_indices, tensor_results, strict=True):
                results[idx] = tensor

        if non_tensor_indices:
            non_tensor_keys = [keys[i] for i in non_tensor_indices]
            non_tensor_results = self._batch_get_bytes(non_tensor_keys)
            for idx, data in zip(non_tensor_indices, non_tensor_results, strict=True):
                results[idx] = pickle.loads(data)

        return results

    def _batch_get_tensors(
        self, keys: list[str], shapes: list, dtypes: list
    ) -> list[Tensor]:
        import numpy as np
        
        all_bytes = []
        
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i:i + BATCH_SIZE_LIMIT]
            batch_results = self._store.get_batch(batch_keys)
            if len(batch_results) != len(batch_keys):
                raise RuntimeError(
                    f"get_batch returned {len(batch_results)} items, expected {len(batch_keys)}"
                )
            all_bytes.extend(batch_results)

        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.float16: np.float16,
            torch.bfloat16: np.int16,
        }

        tensors = [None] * len(keys)
        for i, (raw_bytes, shape, dtype) in enumerate(zip(all_bytes, shapes, dtypes, strict=True)):
            np_dtype = dtype_map.get(dtype, np.float32)
            arr = np.frombuffer(raw_bytes, dtype=np_dtype)
            if dtype == torch.bfloat16:
                tensor = torch.empty(shape, dtype=torch.int16)
                tensor.view(-1).copy_(torch.from_numpy(arr))
                tensor = tensor.view(torch.bfloat16)
            else:
                tensor = torch.empty(shape, dtype=dtype)
                tensor.view(-1).copy_(torch.from_numpy(arr))
            tensors[i] = tensor

        return tensors

    def _batch_get_bytes(self, keys: list[str]) -> list[bytes]:
        results = []
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i:i + BATCH_SIZE_LIMIT]
            batch_results = self._store.get_batch(batch_keys)
            if len(batch_results) != len(batch_keys):
                raise RuntimeError(
                    f"get_batch returned {len(batch_results)} items, expected {len(batch_keys)}"
                )
            results.extend(batch_results)
        return results

    def clear(self, keys: list[str]):
        for key in keys:
            ret = self._store.remove(key)
            if ret != 0:
                logger.warning(f"remove failed for key '{key}' with error code: {ret}")

    def close(self):
        if self._store:
            self._store.close()
            self._store = None

