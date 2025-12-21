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


@StorageClientFactory.register("MooncakeStorageClient")
class MooncakeStorageClient(TransferQueueStorageKVClient):
    """
    Storage client for Mooncake Store.

    Supports storing and fetching both:
    - Tensors via put_tensor/get_tensor (high performance, no pickle overhead).
    - General objects (str, bool, list, etc.) via put/get with pickle serialization.
    """

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

        logger.debug(
            f"MooncakeStorageClient initialized: "
            f"local_hostname={self.local_hostname}, "
            f"metadata_server={self.metadata_server}, "
            f"master_server_address={self.master_server_address}"
        )

    def put(self, keys: list[str], values: list[Any]):
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        total_items = len(keys)
        logger.debug(f"MooncakeStorageClient: Putting {total_items} items")
        
        tensor_items = []
        non_tensor_keys = []
        non_tensor_values_bytes = []
        
        for key, value in zip(keys, values, strict=True):
            if isinstance(value, torch.Tensor):
                tensor_items.append((key, value.contiguous()))
            else:
                non_tensor_keys.append(key)
                non_tensor_values_bytes.append(pickle.dumps(value))
        
        if tensor_items:
            for key, tensor in tensor_items:
                ret = self._store.put_tensor(key, tensor)
                if ret != 0:
                    raise RuntimeError(f"put_tensor failed for key '{key}' with error code: {ret}")
            logger.debug(f"MooncakeStorageClient: Put {len(tensor_items)} tensors via zero-copy put_tensor API")
        
        if non_tensor_keys:
            batch_size = 1000
            for i in range(0, len(non_tensor_keys), batch_size):
                batch_keys = non_tensor_keys[i:i + batch_size]
                batch_values = non_tensor_values_bytes[i:i + batch_size]
                ret = self._store.put_batch(batch_keys, batch_values)
                if ret != 0:
                    raise RuntimeError(
                        f"put_batch failed for non-tensors batch {i//batch_size + 1} "
                        f"(items {i} to {min(i+batch_size, len(non_tensor_keys))}) with error code: {ret}"
                    )
            logger.debug(f"MooncakeStorageClient: Put {len(non_tensor_keys)} non-tensors via batch API")
        
        logger.debug(f"MooncakeStorageClient: Successfully put all {total_items} items")

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Any]:
        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStorageClient needs shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        total_items = len(keys)
        logger.debug(f"MooncakeStorageClient: Getting {total_items} items")
        
        results = []
        tensor_indices = []
        non_tensor_indices = []
        
        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)
        
        tensor_results = {}
        if tensor_indices:
            for i in tensor_indices:
                tensor = self._store.get_tensor(keys[i])
                if tensor is None:
                    raise RuntimeError(f"get_tensor failed for key '{keys[i]}'")
                tensor_results[i] = tensor
            logger.debug(f"MooncakeStorageClient: Got {len(tensor_indices)} tensors via zero-copy get_tensor API")
        
        if non_tensor_indices:
            batch_size = 1000
            for i in range(0, len(non_tensor_indices), batch_size):
                batch_indices = non_tensor_indices[i:i + batch_size]
                batch_keys = [keys[j] for j in batch_indices]
                raw_data_list = self._store.get_batch(batch_keys)
                if len(raw_data_list) != len(batch_keys):
                    raise RuntimeError(
                        f"get_batch returned {len(raw_data_list)} items, expected {len(batch_keys)} "
                        f"for batch {i//batch_size + 1}"
                    )
                for idx, raw_data in zip(batch_indices, raw_data_list, strict=True):
                    if not raw_data:
                        raise RuntimeError(f"get_batch failed for key '{keys[idx]}': empty data")
                    results.append((idx, pickle.loads(raw_data)))
            logger.debug(f"MooncakeStorageClient: Got {len(non_tensor_indices)} non-tensors via batch API")
        
        final_results = [None] * len(keys)
        for i, tensor in tensor_results.items():
            final_results[i] = tensor
        for i, value in results:
            final_results[i] = value
        
        logger.debug(f"MooncakeStorageClient: Successfully got all {total_items} items")
        return final_results
    
    @staticmethod
    def _dtype_to_numpy(dtype):
        import numpy as np
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
        }
        return dtype_map.get(dtype, np.float32)

    def clear(self, keys: list[str]):
        for key in keys:
            ret = self._store.remove(key)
            if ret != 0:
                logger.warning(f"remove failed for key '{key}' with error code: {ret}")

    def close(self):
        if self._store:
            self._store.close()
            self._store = None

