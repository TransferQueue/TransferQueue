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
        put_start_time = time.time()
        
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        classify_start = time.time()
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
        classify_time = time.time() - classify_start

        tensor_time = 0.0
        non_tensor_time = 0.0
        
        if tensor_keys:
            tensor_start = time.time()
            self._batch_put_tensors(tensor_keys, tensor_values)
            tensor_time = time.time() - tensor_start

        if non_tensor_keys:
            non_tensor_start = time.time()
            self._batch_put_bytes(non_tensor_keys, non_tensor_values)
            non_tensor_time = time.time() - non_tensor_start

        total_time = time.time() - put_start_time
        
        logger.warning("=" * 80)
        logger.warning("MooncakeStorageClient: put() Method Summary")
        logger.warning("=" * 80)
        logger.warning(f"Total items: {len(keys)} (tensors: {len(tensor_keys)}, non-tensors: {len(non_tensor_keys)})")
        logger.warning(f"Total time: {total_time:.4f}s")
        logger.warning("Time Breakdown:")
        logger.warning(f"  ├─ Classify items:     {classify_time:8.4f}s ({classify_time/total_time*100:5.1f}%)")
        if tensor_keys:
            logger.warning(f"  ├─ Tensor operations:   {tensor_time:8.4f}s ({tensor_time/total_time*100:5.1f}%)")
        if non_tensor_keys:
            logger.warning(f"  ├─ Non-tensor ops:      {non_tensor_time:8.4f}s ({non_tensor_time/total_time*100:5.1f}%)")
        other_time = total_time - classify_time - tensor_time - non_tensor_time
        if other_time > 0.001:
            logger.warning(f"  └─ Other overhead:      {other_time:8.4f}s ({other_time/total_time*100:5.1f}%)")
        logger.warning("=" * 80)

    def _batch_put_tensors(self, keys: list[str], tensors: list[Tensor]):
        total_serialize_time = 0.0
        total_put_batch_time = 0.0
        total_put_bytes = 0
        
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i:i + BATCH_SIZE_LIMIT]
            batch_tensors = tensors[i:i + BATCH_SIZE_LIMIT]
            
            serialize_start = time.time()
            batch_values = []
            for tensor in batch_tensors:
                if tensor.dtype == torch.bfloat16:
                    bytes_data = tensor.detach().cpu().view(torch.int16).numpy().tobytes()
                else:
                    bytes_data = tensor.detach().cpu().numpy().tobytes()
                batch_values.append(bytes_data)
                total_put_bytes += len(bytes_data)
            total_serialize_time += time.time() - serialize_start
            
            put_batch_start = time.time()
            ret = self._store.put_batch(batch_keys, batch_values)
            total_put_batch_time += time.time() - put_batch_start
            if ret != 0:
                raise RuntimeError(f"put_batch failed with error code: {ret}")

        total_time = total_serialize_time + total_put_batch_time
        put_batch_throughput = (total_put_bytes * 8 / (1024**3)) / total_put_batch_time if total_put_batch_time > 0 else 0
        
        logger.warning("=" * 80)
        logger.warning("MooncakeStorageClient: _batch_put_tensors Time Breakdown")
        logger.warning("=" * 80)
        logger.warning(f"Total tensors: {len(keys)}, Total bytes: {total_put_bytes / (1024**3):.2f} GB")
        logger.warning(f"Total time: {total_time:.4f}s")
        logger.warning("Time Breakdown:")
        logger.warning(f"  ├─ serialize (tensor->bytes): {total_serialize_time:8.4f}s ({total_serialize_time/total_time*100:5.1f}%) "
                      f"[{total_serialize_time/len(keys)*1000:.4f} ms/tensor]")
        logger.warning(f"  └─ put_batch (network):        {total_put_batch_time:8.4f}s ({total_put_batch_time/total_time*100:5.1f}%) "
                      f"[throughput: {put_batch_throughput:.2f} Gb/s]")
        logger.warning("=" * 80)

    def _batch_put_bytes(self, keys: list[str], values: list[bytes]):
        total_put_batch_time = 0.0
        total_put_bytes = sum(len(v) for v in values)
        
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i:i + BATCH_SIZE_LIMIT]
            batch_values = values[i:i + BATCH_SIZE_LIMIT]
            
            put_batch_start = time.time()
            ret = self._store.put_batch(batch_keys, batch_values)
            total_put_batch_time += time.time() - put_batch_start
            if ret != 0:
                raise RuntimeError(f"put_batch failed with error code: {ret}")
        
        put_batch_throughput = (total_put_bytes * 8 / (1024**3)) / total_put_batch_time if total_put_batch_time > 0 else 0
        logger.debug(f"MooncakeStorageClient: _batch_put_bytes - {len(keys)} items, "
                    f"{total_put_bytes / (1024**3):.2f} GB, "
                    f"{total_put_batch_time:.4f}s, "
                    f"throughput: {put_batch_throughput:.2f} Gb/s")

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Any]:
        get_start_time = time.time()
        
        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStorageClient needs shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        classify_start = time.time()
        tensor_indices = []
        non_tensor_indices = []

        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)
        classify_time = time.time() - classify_start

        results = [None] * len(keys)

        tensor_time = 0.0
        non_tensor_time = 0.0
        
        if tensor_indices:
            tensor_keys = [keys[i] for i in tensor_indices]
            tensor_shapes = [shapes[i] for i in tensor_indices]
            tensor_dtypes = [dtypes[i] for i in tensor_indices]
            tensor_start = time.time()
            tensor_results = self._batch_get_tensors(tensor_keys, tensor_shapes, tensor_dtypes)
            for idx, tensor in zip(tensor_indices, tensor_results, strict=True):
                results[idx] = tensor
            tensor_time = time.time() - tensor_start

        if non_tensor_indices:
            non_tensor_keys = [keys[i] for i in non_tensor_indices]
            non_tensor_start = time.time()
            non_tensor_results = self._batch_get_bytes(non_tensor_keys)
            for idx, data in zip(non_tensor_indices, non_tensor_results, strict=True):
                results[idx] = pickle.loads(data)
            non_tensor_time = time.time() - non_tensor_start

        total_time = time.time() - get_start_time
        
        logger.warning("=" * 80)
        logger.warning("MooncakeStorageClient: get() Method Summary")
        logger.warning("=" * 80)
        logger.warning(f"Total items: {len(keys)} (tensors: {len(tensor_indices)}, non-tensors: {len(non_tensor_indices)})")
        logger.warning(f"Total time: {total_time:.4f}s")
        logger.warning("Time Breakdown:")
        logger.warning(f"  ├─ Classify items:     {classify_time:8.4f}s ({classify_time/total_time*100:5.1f}%)")
        if tensor_indices:
            logger.warning(f"  ├─ Tensor operations:   {tensor_time:8.4f}s ({tensor_time/total_time*100:5.1f}%)")
        if non_tensor_indices:
            logger.warning(f"  ├─ Non-tensor ops:      {non_tensor_time:8.4f}s ({non_tensor_time/total_time*100:5.1f}%)")
        other_time = total_time - classify_time - tensor_time - non_tensor_time
        if other_time > 0.001:
            logger.warning(f"  └─ Other overhead:      {other_time:8.4f}s ({other_time/total_time*100:5.1f}%)")
        logger.warning("=" * 80)

        return results

    def _batch_get_tensors(
        self, keys: list[str], shapes: list, dtypes: list
    ) -> list[Tensor]:
        tensors = [None] * len(keys)
        
        total_get_batch_time = 0.0
        total_frombuffer_time = 0.0
        total_get_batch_bytes = 0
            
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i:i + BATCH_SIZE_LIMIT]
            batch_shapes = shapes[i:i + BATCH_SIZE_LIMIT]
            batch_dtypes = dtypes[i:i + BATCH_SIZE_LIMIT]
            
            get_batch_start = time.time()
            batch_results = self._store.get_batch(batch_keys)
            total_get_batch_time += time.time() - get_batch_start
            
            if len(batch_results) != len(batch_keys):
                        raise RuntimeError(
                    f"get_batch returned {len(batch_results)} items, expected {len(batch_keys)}"
                )
            
            frombuffer_start = time.time()
            for j, (raw_bytes, shape, dtype) in enumerate(zip(batch_results, batch_shapes, batch_dtypes, strict=True)):
                total_get_batch_bytes += len(raw_bytes)
                if dtype == torch.bfloat16:
                    tensors[i + j] = torch.frombuffer(raw_bytes, dtype=torch.int16).view(shape).view(torch.bfloat16)
                        else:
                    tensors[i + j] = torch.frombuffer(raw_bytes, dtype=dtype).view(shape)
            total_frombuffer_time += time.time() - frombuffer_start

        total_time = total_get_batch_time + total_frombuffer_time
        get_batch_throughput = (total_get_batch_bytes * 8 / (1024**3)) / total_get_batch_time if total_get_batch_time > 0 else 0
        
        logger.warning("=" * 80)
        logger.warning("MooncakeStorageClient: _batch_get_tensors Time Breakdown")
        logger.warning("=" * 80)
        logger.warning(f"Total tensors: {len(keys)}, Total bytes: {total_get_batch_bytes / (1024**3):.2f} GB")
        logger.warning(f"Total time: {total_time:.4f}s")
        logger.warning("Time Breakdown:")
        logger.warning(f"  ├─ get_batch (network):    {total_get_batch_time:8.4f}s ({total_get_batch_time/total_time*100:5.1f}%) "
                      f"[throughput: {get_batch_throughput:.2f} Gb/s]")
        logger.warning(f"  └─ frombuffer (deserialize): {total_frombuffer_time:8.4f}s ({total_frombuffer_time/total_time*100:5.1f}%) "
                      f"[{total_frombuffer_time/len(keys)*1000:.4f} ms/tensor]")
        logger.warning("=" * 80)

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

