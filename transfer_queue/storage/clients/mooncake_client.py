import logging
import os
import pickle
import time
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
        batch_size = 1000
        logger.debug(f"MooncakeStorageClient: Putting {total_items} items using zero-copy batch_put_from")
        
        tensor_items = []
        non_tensor_keys = []
        non_tensor_values_bytes = []
        
        for key, value in zip(keys, values, strict=True):
            if isinstance(value, torch.Tensor):
                tensor = value.contiguous()
                tensor_items.append((key, tensor))
            else:
                non_tensor_keys.append(key)
                non_tensor_values_bytes.append(pickle.dumps(value))
        
        if tensor_items:
            tensor_start_time = time.time()
            for i in range(0, len(tensor_items), batch_size):
                batch_items = tensor_items[i:i + batch_size]
                batch_keys = [item[0] for item in batch_items]
                batch_tensors = [item[1] for item in batch_items]
                
                batch_values_bytes = []
                
                import numpy as np
                metadata_size = 24
                
                for tensor in batch_tensors:
                    dtype_enum = self._dtype_to_tensor_dtype_enum(tensor.dtype)
                    if dtype_enum is None:
                        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
                    
                    ndim = len(tensor.shape)
                    if ndim > 4:
                        raise ValueError(f"Tensor has more than 4 dimensions: {ndim}")
                    
                    metadata = np.zeros(6, dtype=np.int32)
                    metadata[0] = dtype_enum
                    metadata[1] = ndim
                    for j in range(4):
                        if j < ndim:
                            metadata[2 + j] = tensor.shape[j]
                        else:
                            metadata[2 + j] = -1
                    
                    tensor_np = tensor.detach().cpu().numpy()
                    tensor_bytes = tensor_np.tobytes()
                    metadata_bytes = metadata.tobytes()
                    
                    combined_bytes = metadata_bytes + tensor_bytes
                    batch_values_bytes.append(combined_bytes)
                
                ret = self._store.put_batch(batch_keys, batch_values_bytes)
                if ret != 0:
                    raise RuntimeError(
                        f"put_batch failed for tensors batch {i//batch_size + 1} "
                        f"(items {i} to {min(i+batch_size, len(tensor_items))}) with error code: {ret}"
                    )
                
                if (i + 1) % (batch_size * 10) == 0 or i + batch_size >= len(tensor_items):
                    logger.debug(
                        f"MooncakeStorageClient: Put {min(i + batch_size, len(tensor_items))}/{len(tensor_items)} tensors "
                        f"via put_batch API"
                    )
            
            tensor_end_time = time.time()
            tensor_elapsed = tensor_end_time - tensor_start_time
            logger.info(
                f"MooncakeStorageClient: Put {len(tensor_items)} tensors "
                f"via put_batch in {len(tensor_items)//batch_size + 1} batches, "
                f"cost time: {tensor_elapsed:.8f}s"
            )
        
        if non_tensor_keys:
            non_tensor_start_time = time.time()
            for i in range(0, len(non_tensor_keys), batch_size):
                batch_keys = non_tensor_keys[i:i + batch_size]
                batch_values = non_tensor_values_bytes[i:i + batch_size]
                ret = self._store.put_batch(batch_keys, batch_values)
                if ret != 0:
                    raise RuntimeError(
                        f"put_batch failed for non-tensors batch {i//batch_size + 1} "
                        f"(items {i} to {min(i+batch_size, len(non_tensor_keys))}) with error code: {ret}"
                    )
            non_tensor_end_time = time.time()
            non_tensor_elapsed = non_tensor_end_time - non_tensor_start_time
            logger.info(
                f"MooncakeStorageClient: Put {len(non_tensor_keys)} non-tensors via batch API, "
                f"cost time: {non_tensor_elapsed:.8f}s"
            )
        
        logger.debug(f"MooncakeStorageClient: Successfully put all {total_items} items")

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Any]:
        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStorageClient needs shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        total_items = len(keys)
        batch_size = 1000
        logger.debug(f"MooncakeStorageClient: Getting {total_items} items using zero-copy batch_get_into")
        
        tensor_indices = []
        non_tensor_indices = []
        
        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)
        
        final_results = [None] * len(keys)
        
        if tensor_indices:
            for i in range(0, len(tensor_indices), batch_size):
                batch_indices = tensor_indices[i:i + batch_size]
                batch_keys = [keys[j] for j in batch_indices]
                batch_shapes = [shapes[j] for j in batch_indices]
                batch_dtypes = [dtypes[j] for j in batch_indices]
                
                raw_data_list = self._store.get_batch(batch_keys)
                if len(raw_data_list) != len(batch_keys):
                    raise RuntimeError(
                        f"get_batch returned {len(raw_data_list)} items, expected {len(batch_keys)} "
                        f"for batch {i//batch_size + 1}"
                    )
                
                import numpy as np
                metadata_size = 24
                
                for idx, raw_data, shape, dtype in zip(
                    batch_indices, raw_data_list, batch_shapes, batch_dtypes, strict=True
                ):
                    if not raw_data:
                        raise RuntimeError(f"get_batch failed for key '{batch_keys[batch_indices.index(idx)]}': empty data")
                    
                    if len(raw_data) < metadata_size:
                        raise RuntimeError(
                            f"get_batch returned insufficient data for key '{batch_keys[batch_indices.index(idx)]}': "
                            f"got {len(raw_data)} bytes, expected at least {metadata_size} bytes"
                        )
                    
                    tensor_data = raw_data[metadata_size:]
                    np_dtype = self._dtype_to_numpy(dtype)
                    tensor_array = np.frombuffer(tensor_data, dtype=np_dtype)
                    if shape:
                        tensor_array = tensor_array.reshape(shape)
                    tensor = torch.from_numpy(tensor_array).clone()
                    final_results[idx] = tensor
                
                if (i + 1) % (batch_size * 10) == 0 or i + batch_size >= len(tensor_indices):
                    logger.debug(
                        f"MooncakeStorageClient: Got {min(i + batch_size, len(tensor_indices))}/{len(tensor_indices)} tensors "
                        f"via get_batch API"
                    )
            
            logger.debug(
                f"MooncakeStorageClient: Got {len(tensor_indices)} tensors "
                f"via get_batch in {len(tensor_indices)//batch_size + 1} batches"
            )
        
        if non_tensor_indices:
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
                    final_results[idx] = pickle.loads(raw_data)
            logger.debug(f"MooncakeStorageClient: Got {len(non_tensor_indices)} non-tensors via batch API")
        
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
    
    @staticmethod
    def _dtype_to_tensor_dtype_enum(dtype):
        dtype_map = {
            torch.float32: 0,
            torch.float64: 1,
            torch.int8: 2,
            torch.uint8: 3,
            torch.int16: 4,
            torch.int32: 5,
            torch.int64: 6,
            torch.bool: 7,
            torch.float16: 8,
            torch.bfloat16: 9,
        }
        return dtype_map.get(dtype)

    def clear(self, keys: list[str]):
        for key in keys:
            ret = self._store.remove(key)
            if ret != 0:
                logger.warning(f"remove failed for key '{key}' with error code: {ret}")

    def close(self):
        if self._store:
            self._store.close()
            self._store = None

