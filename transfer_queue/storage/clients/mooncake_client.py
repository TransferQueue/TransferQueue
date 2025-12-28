import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        initial_batch_size = 200
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
            batch_size = initial_batch_size
            i = 0
            total_put_batch_time = 0.0
            total_put_bytes = 0
            
            while i < len(tensor_items):
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
                    total_put_bytes += len(combined_bytes)
                
                put_batch_start = time.time()
                ret = self._store.put_batch(batch_keys, batch_values_bytes)
                put_batch_time = time.time() - put_batch_start
                total_put_batch_time += put_batch_time
                
                if ret != 0:
                    if ret == -600 and batch_size > 1:
                        new_batch_size = max(1, batch_size // 2)
                        logger.warning(
                            f"put_batch failed with buffer allocation error (code: {ret}), "
                            f"reducing batch size from {batch_size} to {new_batch_size}"
                        )
                        batch_size = new_batch_size
                        continue
                    else:
                        raise RuntimeError(
                            f"put_batch failed for tensors batch starting at item {i} "
                            f"(batch_size={batch_size}) with error code: {ret}"
                        )
                
                i += batch_size
                if i % (initial_batch_size * 10) == 0 or i >= len(tensor_items):
                    logger.debug(
                        f"MooncakeStorageClient: Put {min(i, len(tensor_items))}/{len(tensor_items)} tensors "
                        f"via put_batch API"
                    )
            
            tensor_end_time = time.time()
            tensor_elapsed = tensor_end_time - tensor_start_time
            put_batch_throughput_gbps = (total_put_bytes * 8 / (1024**3)) / total_put_batch_time if total_put_batch_time > 0 else 0
            logger.warning(
                f"MooncakeStorageClient: Put {len(tensor_items)} tensors "
                f"via put_batch, cost time: {tensor_elapsed:.8f}s, "
                f"put_batch time: {total_put_batch_time:.8f}s ({total_put_batch_time/tensor_elapsed*100:.1f}%), "
                f"put_batch throughput: {put_batch_throughput_gbps:.2f} Gb/s, "
                f"total data: {total_put_bytes / (1024**3):.2f} GB"
            )
        
        if non_tensor_keys:
            non_tensor_start_time = time.time()
            batch_size = initial_batch_size
            i = 0
            while i < len(non_tensor_keys):
                batch_keys = non_tensor_keys[i:i + batch_size]
                batch_values = non_tensor_values_bytes[i:i + batch_size]
                ret = self._store.put_batch(batch_keys, batch_values)
                if ret != 0:
                    if ret == -600 and batch_size > 1:
                        new_batch_size = max(1, batch_size // 2)
                        logger.warning(
                            f"put_batch failed with buffer allocation error (code: {ret}), "
                            f"reducing batch size from {batch_size} to {new_batch_size}"
                        )
                        batch_size = new_batch_size
                        continue
                    else:
                        raise RuntimeError(
                            f"put_batch failed for non-tensors batch starting at item {i} "
                            f"(batch_size={batch_size}) with error code: {ret}"
                        )
                i += batch_size
            non_tensor_end_time = time.time()
            non_tensor_elapsed = non_tensor_end_time - non_tensor_start_time
            logger.warning(
                f"MooncakeStorageClient: Put {len(non_tensor_keys)} non-tensors via batch API, "
                f"cost time: {non_tensor_elapsed:.8f}s"
            )
        
        logger.debug(f"MooncakeStorageClient: Successfully put all {total_items} items")

    def _process_tensor_batch(self, batch_info):
        """Process a single batch of tensors. Used for parallel batch processing."""
        batch_idx, batch_indices, batch_keys, batch_shapes, batch_dtypes, keys = batch_info
        metadata_size = 24
        
        # Get batch data
        get_batch_start = time.time()
        raw_data_list = self._store.get_batch(batch_keys)
        get_batch_time = time.time() - get_batch_start
        
        # Calculate data size for throughput calculation
        get_batch_bytes = sum(len(raw_data) for raw_data in raw_data_list if raw_data)
        
        # Check if batch size needs to be reduced
        if len(raw_data_list) != len(batch_keys):
            return {
                'batch_idx': batch_idx,
                'success': False,
                'needs_retry': True,
                'get_batch_time': get_batch_time,
                'tensor_convert_time': 0.0,
                'get_batch_bytes': 0,
                'results': None
            }
        
        # Convert tensors
        # Reference: YuanrongClient approach - trust business layer, no validation
        # Only do essential operations: remove metadata header and group by dtype
        tensor_convert_start = time.time()
        batch_results = {}
        
        # Pre-compute element sizes by dtype to avoid repeated calculations
        dtype_element_sizes = {}
        
        # Group by dtype for batch processing (similar to YuanrongClient's approach)
        # Trust business layer: use provided shapes and dtypes directly
        dtype_groups = {}
        
        # Performance monitoring: group phase
        group_start = time.time()
        for idx, raw_data, shape, dtype in zip(
            batch_indices, raw_data_list, batch_shapes, batch_dtypes, strict=True
        ):
            # Trust storage layer: assume raw_data is valid
            # Only extract tensor_data (remove 24-byte metadata header)
            tensor_data = raw_data[metadata_size:] if raw_data else None
            
            if not tensor_data:
                # Handle empty tensor case
                if shape and 0 in shape:
                    batch_results[idx] = torch.empty(shape, dtype=dtype)
                continue
            
            # Cache element_size calculation per dtype
            if dtype not in dtype_element_sizes:
                dtype_element_sizes[dtype] = torch.tensor(0, dtype=dtype).element_size()
            
            # Group by dtype for batch processing
            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append((idx, tensor_data, shape))
        group_time = time.time() - group_start
        
        # Batch process tensors by dtype (grouped processing reduces overhead)
        # Reference: YuanrongClient - directly use provided shapes and dtypes, no shape checking
        frombuffer_time = 0.0
        view_time = 0.0
        slice_time = 0.0
        skipped_view_count = 0
        
        # Process tensors by dtype (similar to YuanrongClient's approach)
        for dtype, items in dtype_groups.items():
            element_size = dtype_element_sizes[dtype]
            
            for idx, tensor_data, shape in items:
                if not tensor_data:
                    continue
                
                # Performance monitoring: torch.frombuffer time
                frombuffer_start = time.time()
                tensor = torch.frombuffer(tensor_data, dtype=dtype)
                frombuffer_time += time.time() - frombuffer_start
                
                # Trust business layer: use provided shape directly (like YuanrongClient)
                if shape:
                    # Check if 1D tensor that can skip view (optimization)
                    if len(shape) == 1:
                        # For 1D tensors, if shape matches implicit 1D shape, skip view
                        expected_elements = shape[0]
                        actual_elements = len(tensor_data) // element_size
                        if expected_elements == actual_elements:
                            # Skip view for 1D tensors with matching shape
                            skipped_view_count += 1
                        else:
                            # Need to reshape (trust business layer shape)
                            view_start = time.time()
                            tensor = tensor.view(shape)
                            view_time += time.time() - view_start
                    else:
                        # Multi-dimensional tensor, need view (trust business layer shape)
                        view_start = time.time()
                        tensor = tensor.view(shape)
                        view_time += time.time() - view_start
                
                batch_results[idx] = tensor
        
        empty_tensor_time = 0.0
        
        tensor_convert_time = time.time() - tensor_convert_start
        
        # Calculate other overhead (total - measured components)
        other_overhead = tensor_convert_time - group_time - frombuffer_time - view_time - slice_time - empty_tensor_time
        
        # Store detailed timing for analysis
        tensor_convert_details = {
            'group_time': group_time,
            'frombuffer_time': frombuffer_time,
            'view_time': view_time,
            'slice_time': slice_time,
            'empty_tensor_time': empty_tensor_time,
            'other_overhead': other_overhead,
            'total_tensors': len(batch_results),
            'num_frombuffer_calls': sum(len(items) for items in dtype_groups.values()),
            'skipped_view_count': skipped_view_count
        }
        
        return {
            'batch_idx': batch_idx,
            'success': True,
            'needs_retry': False,
            'get_batch_time': get_batch_time,
            'tensor_convert_time': tensor_convert_time,
            'tensor_convert_details': tensor_convert_details,
            'get_batch_bytes': get_batch_bytes,
            'results': batch_results
        }

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Any]:
        get_method_start_time = time.time()
        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStorageClient needs shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        total_items = len(keys)
        initial_batch_size = 1000  # Increase batch size to reduce number of batches
        logger.debug(f"MooncakeStorageClient: Getting {total_items} items using zero-copy batch_get_into")
        
        classify_start_time = time.time()
        tensor_indices = []
        non_tensor_indices = []
        
        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)
        
        final_results = [None] * len(keys)
        classify_time = time.time() - classify_start_time
        
        if tensor_indices:
            get_start_time = time.time()
            batch_size = initial_batch_size
            total_get_batch_time_sum = 0.0  # Sum of all batch get_batch times (for reference)
            max_get_batch_time = 0.0  # Max get_batch time for parallel processing
            max_tensor_convert_time = 0.0  # Track max instead of sum for parallel processing
            total_tensor_convert_time_sum = 0.0  # Sum for reference
            total_get_batch_bytes = 0  # Total bytes retrieved via get_batch
            
            # Aggregate detailed tensor conversion timing
            total_group_time = 0.0
            total_frombuffer_time = 0.0
            total_view_time = 0.0
            total_slice_time = 0.0
            total_empty_tensor_time = 0.0
            total_other_overhead = 0.0
            total_tensor_count = 0
            total_frombuffer_calls = 0
            total_skipped_view_count = 0
            
            # Prepare all batches
            batches = []
            i = 0
            batch_idx = 0
            while i < len(tensor_indices):
                batch_indices = tensor_indices[i:i + batch_size]
                batch_keys = [keys[j] for j in batch_indices]
                batch_shapes = [shapes[j] for j in batch_indices]
                batch_dtypes = [dtypes[j] for j in batch_indices]
                batches.append((batch_idx, batch_indices, batch_keys, batch_shapes, batch_dtypes, keys))
                i += batch_size
                batch_idx += 1
            
            # Process batches in parallel with adaptive retry
            max_workers = min(16, len(batches))  # Increase concurrent batches for better parallelism
            retry_batches = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._process_tensor_batch, batch_info): batch_info
                    for batch_info in batches
                }
                
                # Collect results
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        result = future.result()
                        if result['success']:
                            # Update final_results
                            if result['results']:
                                for idx, tensor in result['results'].items():
                                    final_results[idx] = tensor
                            total_get_batch_time_sum += result['get_batch_time']
                            max_get_batch_time = max(max_get_batch_time, result['get_batch_time'])
                            total_tensor_convert_time_sum += result['tensor_convert_time']
                            max_tensor_convert_time = max(max_tensor_convert_time, result['tensor_convert_time'])
                            total_get_batch_bytes += result.get('get_batch_bytes', 0)
                            
                            # Aggregate detailed timing
                            details = result.get('tensor_convert_details', {})
                            total_group_time += details.get('group_time', 0.0)
                            total_frombuffer_time += details.get('frombuffer_time', 0.0)
                            total_view_time += details.get('view_time', 0.0)
                            total_slice_time += details.get('slice_time', 0.0)
                            total_empty_tensor_time += details.get('empty_tensor_time', 0.0)
                            total_other_overhead += details.get('other_overhead', 0.0)
                            total_tensor_count += details.get('total_tensors', 0)
                            total_frombuffer_calls += details.get('num_frombuffer_calls', 0)
                            total_skipped_view_count += details.get('skipped_view_count', 0)
                        else:
                            # Need retry with smaller batch size
                            retry_batches.append(batch_info)
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_info[0]}: {e}")
                        retry_batches.append(batch_info)
            
            # Retry failed batches with smaller batch size
            retry_batch_size = batch_size
            while retry_batches:
                retry_batch_size = max(1, retry_batch_size // 2)
                if retry_batch_size == 1:
                    # Last resort: process sequentially
                    for batch_info in retry_batches:
                        result = self._process_tensor_batch(batch_info)
                        if result['success']:
                            if result['results']:
                                for idx, tensor in result['results'].items():
                                    final_results[idx] = tensor
                            total_get_batch_time_sum += result['get_batch_time']
                            max_get_batch_time = max(max_get_batch_time, result['get_batch_time'])
                            total_tensor_convert_time_sum += result['tensor_convert_time']
                            max_tensor_convert_time = max(max_tensor_convert_time, result['tensor_convert_time'])
                            total_get_batch_bytes += result.get('get_batch_bytes', 0)
                            
                            # Aggregate detailed timing
                            details = result.get('tensor_convert_details', {})
                            total_group_time += details.get('group_time', 0.0)
                            total_frombuffer_time += details.get('frombuffer_time', 0.0)
                            total_view_time += details.get('view_time', 0.0)
                            total_slice_time += details.get('slice_time', 0.0)
                            total_empty_tensor_time += details.get('empty_tensor_time', 0.0)
                            total_other_overhead += details.get('other_overhead', 0.0)
                            total_tensor_count += details.get('total_tensors', 0)
                            total_frombuffer_calls += details.get('num_frombuffer_calls', 0)
                            total_skipped_view_count += details.get('skipped_view_count', 0)
                        else:
                            raise RuntimeError(f"Failed to process batch {batch_info[0]} even with batch_size=1")
                    break
                
                logger.warning(
                    f"Retrying {len(retry_batches)} batches with reduced batch_size={retry_batch_size}"
                )
                
                # Split retry batches into smaller batches
                new_retry_batches = []
                for batch_info in retry_batches:
                    batch_idx, batch_indices, batch_keys, batch_shapes, batch_dtypes, keys = batch_info
                    # Split into smaller batches
                    for j in range(0, len(batch_indices), retry_batch_size):
                        sub_batch_indices = batch_indices[j:j + retry_batch_size]
                        sub_batch_keys = [keys[idx] for idx in sub_batch_indices]
                        sub_batch_shapes = [shapes[idx] for idx in sub_batch_indices]
                        sub_batch_dtypes = [dtypes[idx] for idx in sub_batch_indices]
                        new_retry_batches.append((
                            batch_idx * 1000 + j,  # Unique batch idx
                            sub_batch_indices,
                            sub_batch_keys,
                            sub_batch_shapes,
                            sub_batch_dtypes,
                            keys
                        ))
                
                retry_batches = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self._process_tensor_batch, batch_info): batch_info
                        for batch_info in new_retry_batches
                    }
                    
                    for future in as_completed(future_to_batch):
                        batch_info = future_to_batch[future]
                        try:
                            result = future.result()
                            if result['success']:
                                if result['results']:
                                    for idx, tensor in result['results'].items():
                                        final_results[idx] = tensor
                                total_get_batch_time_sum += result['get_batch_time']
                                max_get_batch_time = max(max_get_batch_time, result['get_batch_time'])
                                total_tensor_convert_time_sum += result['tensor_convert_time']
                                max_tensor_convert_time = max(max_tensor_convert_time, result['tensor_convert_time'])
                                total_get_batch_bytes += result.get('get_batch_bytes', 0)
                                
                                # Aggregate detailed timing
                                details = result.get('tensor_convert_details', {})
                                total_validate_group_time += details.get('validate_group_time', 0.0)
                                total_frombuffer_time += details.get('frombuffer_time', 0.0)
                                total_view_time += details.get('view_time', 0.0)
                                total_slice_time += details.get('slice_time', 0.0)
                                total_empty_tensor_time += details.get('empty_tensor_time', 0.0)
                                total_other_overhead += details.get('other_overhead', 0.0)
                                total_tensor_count += details.get('total_tensors', 0)
                                total_frombuffer_calls += details.get('num_frombuffer_calls', 0)
                                total_skipped_view_count += details.get('skipped_view_count', 0)
                            else:
                                retry_batches.append(batch_info)
                        except Exception as e:
                            logger.error(f"Error processing retry batch {batch_info[0]}: {e}")
                            retry_batches.append(batch_info)
            
            get_end_time = time.time()
            get_elapsed = get_end_time - get_start_time
            # Calculate get_batch throughput
            get_batch_throughput_gbps = (total_get_batch_bytes * 8 / (1024**3)) / max_get_batch_time if max_get_batch_time > 0 else 0
            
            # Calculate average per-tensor times (using sum times for accurate averages)
            avg_frombuffer_time = total_frombuffer_time / total_frombuffer_calls if total_frombuffer_calls > 0 else 0
            avg_view_time = total_view_time / total_frombuffer_calls if total_frombuffer_calls > 0 else 0
            
            # Calculate statistics
            skip_view_ratio = (total_skipped_view_count / total_frombuffer_calls * 100) if total_frombuffer_calls > 0 else 0
            
            # Improved logging format for better readability
            logger.warning("=" * 80)
            logger.warning(f"MooncakeStorageClient: GET Tensor Operation Time Distribution")
            logger.warning("=" * 80)
            logger.warning(f"Total tensors: {len(tensor_indices)}, Total time: {get_elapsed:.4f}s")
            logger.warning(f"Data transferred: {total_get_batch_bytes / (1024**3):.2f} GB, Throughput: {get_batch_throughput_gbps:.2f} Gb/s")
            logger.warning("")
            logger.warning("Time Breakdown (using max time for parallel operations):")
            logger.warning(f"  ├─ Network (get_batch):     {max_get_batch_time:8.4f}s ({max_get_batch_time/get_elapsed*100:5.1f}%)")
            logger.warning(f"  └─ Tensor Conversion:       {max_tensor_convert_time:8.4f}s ({max_tensor_convert_time/get_elapsed*100:5.1f}%)")
            # Calculate proportional times for tensor conversion sub-components
            # Use sum times to estimate proportions, then scale to max time
            tensor_convert_sum = total_group_time + total_frombuffer_time + total_view_time + total_slice_time + total_empty_tensor_time + total_other_overhead
            if tensor_convert_sum > 0 and max_tensor_convert_time > 0:
                scale_factor = max_tensor_convert_time / tensor_convert_sum
                estimated_group = total_group_time * scale_factor
                estimated_frombuffer = total_frombuffer_time * scale_factor
                estimated_view = total_view_time * scale_factor
                estimated_slice = total_slice_time * scale_factor
                estimated_empty = total_empty_tensor_time * scale_factor
                estimated_other = total_other_overhead * scale_factor
                
                logger.warning(f"      ├─ group (slice+group):  {estimated_group:8.4f}s ({estimated_group/max_tensor_convert_time*100:5.1f}%)")
                logger.warning(f"      ├─ frombuffer:            {estimated_frombuffer:8.4f}s ({estimated_frombuffer/max_tensor_convert_time*100:5.1f}%) "
                              f"[avg: {avg_frombuffer_time*1000:.3f}ms/call, {total_frombuffer_calls} calls]")
                logger.warning(f"      ├─ view:                  {estimated_view:8.4f}s ({estimated_view/max_tensor_convert_time*100:5.1f}%) "
                              f"[avg: {avg_view_time*1000:.3f}ms/call, {total_frombuffer_calls - total_skipped_view_count} calls]")
                logger.warning(f"      ├─ slice:                 {estimated_slice:8.4f}s ({estimated_slice/max_tensor_convert_time*100:5.1f}%)")
                logger.warning(f"      ├─ empty_tensor:          {estimated_empty:8.4f}s ({estimated_empty/max_tensor_convert_time*100:5.1f}%)")
                logger.warning(f"      └─ other_overhead:        {estimated_other:8.4f}s ({estimated_other/max_tensor_convert_time*100:5.1f}%)")
            else:
                logger.warning(f"      ├─ group (slice+group):   {0.0:8.4f}s ({0.0:5.1f}%)")
                logger.warning(f"      ├─ frombuffer:            {0.0:8.4f}s ({0.0:5.1f}%)")
                logger.warning(f"      ├─ view:                  {0.0:8.4f}s ({0.0:5.1f}%)")
                logger.warning(f"      ├─ slice:                 {0.0:8.4f}s ({0.0:5.1f}%)")
                logger.warning(f"      ├─ empty_tensor:          {0.0:8.4f}s ({0.0:5.1f}%)")
                logger.warning(f"      └─ other_overhead:        {0.0:8.4f}s ({0.0:5.1f}%)")
            logger.warning("")
            logger.warning(f"Skipped view: {total_skipped_view_count} ({skip_view_ratio:.1f}%)")
            logger.warning("")
            logger.warning("Note: Sum times exceed total due to parallel processing (sum: {:.4f}s)".format(
                total_get_batch_time_sum + total_tensor_convert_time_sum
            ))
            logger.warning("=" * 80)
        
        non_tensor_get_batch_time = 0.0
        non_tensor_pickle_time = 0.0
        if non_tensor_indices:
            non_tensor_start_time = time.time()
            batch_size = initial_batch_size
            i = 0
            while i < len(non_tensor_indices):
                batch_indices = non_tensor_indices[i:i + batch_size]
                batch_keys = [keys[j] for j in batch_indices]
                
                non_tensor_get_batch_start = time.time()
                raw_data_list = self._store.get_batch(batch_keys)
                non_tensor_get_batch_time += time.time() - non_tensor_get_batch_start
                
                if len(raw_data_list) != len(batch_keys):
                    if batch_size > 1:
                        new_batch_size = max(1, batch_size // 2)
                        logger.warning(
                            f"get_batch returned {len(raw_data_list)} items, expected {len(batch_keys)}, "
                            f"reducing batch size from {batch_size} to {new_batch_size}"
                        )
                        batch_size = new_batch_size
                        continue
                    else:
                        raise RuntimeError(
                            f"get_batch returned {len(raw_data_list)} items, expected {len(batch_keys)} "
                            f"for batch starting at index {i}"
                        )
                
                failed_count = 0
                pickle_start = time.time()
                for idx, raw_data in zip(batch_indices, raw_data_list, strict=True):
                    if not raw_data:
                        failed_count += 1
                        if batch_size > 1:
                            break
                        else:
                            raise RuntimeError(f"get_batch failed for key '{keys[idx]}': empty data")
                    final_results[idx] = pickle.loads(raw_data)
                non_tensor_pickle_time += time.time() - pickle_start
                
                if failed_count > 0 and batch_size > 1:
                    new_batch_size = max(1, batch_size // 2)
                    logger.warning(
                        f"get_batch failed for {failed_count} items due to buffer allocation, "
                        f"reducing batch size from {batch_size} to {new_batch_size}"
                    )
                    batch_size = new_batch_size
                    continue
                
                i += batch_size
            non_tensor_total_time = time.time() - non_tensor_start_time
            logger.warning("=" * 80)
            logger.warning(f"MooncakeStorageClient: GET Non-Tensor Operation Time Distribution")
            logger.warning("=" * 80)
            logger.warning(f"Total non-tensors: {len(non_tensor_indices)}, Total time: {non_tensor_total_time:.4f}s")
            logger.warning("Time Breakdown:")
            logger.warning(f"  ├─ Network (get_batch): {non_tensor_get_batch_time:8.4f}s ({non_tensor_get_batch_time/non_tensor_total_time*100:5.1f}%)")
            logger.warning(f"  └─ Deserialize (pickle): {non_tensor_pickle_time:8.4f}s ({non_tensor_pickle_time/non_tensor_total_time*100:5.1f}%)")
            logger.warning("=" * 80)
        
        get_method_end_time = time.time()
        get_method_total_time = get_method_end_time - get_method_start_time
        
        # Calculate component times for summary
        tensor_time = get_elapsed if tensor_indices else 0.0
        non_tensor_time = non_tensor_total_time if non_tensor_indices else 0.0
        
        logger.warning("=" * 80)
        logger.warning(f"MooncakeStorageClient: GET Method Summary")
        logger.warning("=" * 80)
        logger.warning(f"Total items: {total_items} (tensors: {len(tensor_indices) if tensor_indices else 0}, "
                      f"non-tensors: {len(non_tensor_indices) if non_tensor_indices else 0})")
        logger.warning(f"Total time: {get_method_total_time:.4f}s")
        logger.warning("Time Breakdown:")
        logger.warning(f"  ├─ Classify items:     {classify_time:8.4f}s ({classify_time/get_method_total_time*100:5.1f}%)")
        if tensor_indices:
            logger.warning(f"  ├─ Tensor operations:   {tensor_time:8.4f}s ({tensor_time/get_method_total_time*100:5.1f}%)")
        if non_tensor_indices:
            logger.warning(f"  ├─ Non-tensor ops:      {non_tensor_time:8.4f}s ({non_tensor_time/get_method_total_time*100:5.1f}%)")
        other_time = get_method_total_time - classify_time - tensor_time - non_tensor_time
        if other_time > 0.001:
            logger.warning(f"  └─ Other overhead:      {other_time:8.4f}s ({other_time/get_method_total_time*100:5.1f}%)")
        logger.warning("=" * 80)
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

