# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
import zmq
from ray.util import get_node_ip_address

from transfer_queue.metadata import (
    BatchMeta,
    FieldMeta,
    SampleMeta,
)
from transfer_queue.sampler import BaseSampler, SequentialSampler
from transfer_queue.utils.utils import (
    ProductionStatus,
    TransferQueueRole,
)
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

TQ_CONTROLLER_GET_METADATA_TIMEOUT = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_TIMEOUT", 1))
TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL", 0.2))
TQ_CONTROLLER_CONNECTION_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_CONNECTION_CHECK_INTERVAL", 2))

TQ_INIT_SAMPLE_NUM = int(os.environ.get("TQ_INIT_SAMPLE_NUM", 10))  # Initial number of samples
TQ_INIT_FIELD_NUM = int(os.environ.get("TQ_INIT_FIELD_NUM", 10))

# Expansion configuration - Unified approach using minimum expansion sizes
TQ_SAMPLE_MIN_EXPANSION_SIZE = int(
    os.environ.get("TQ_SAMPLE_MIN_EXPANSION_SIZE", 10)
)  # Minimum expansion size for samples (rows)
TQ_FIELD_MIN_EXPANSION_SIZE = int(
    os.environ.get("TQ_FIELD_MIN_EXPANSION_SIZE", 5)
)  # Minimum expansion size for fields (columns)


class PartitionIndexManager:
    """
    Manages the mapping relationship between partitions and global indexes,
    responsible for index allocation and reuse.
    """

    def __init__(self):
        # Records the set of global_indexes used by each partition
        self.partition_to_indexes = defaultdict(set)

        # Reusable global_index pool - stored using list
        self.reusable_indexes = []

        # Global index counter for allocating new indexes
        self.global_index_counter = 0

        # Track all active indexes
        self.allocated_indexes = set()

    def allocate_indexes(self, partition_id, count=1) -> list:
        """
        Allocate global_indexes for the specified partition.
        Prioritizes obtaining from reusable pool, allocates new indexes when insufficient.

        Args:
            partition_id: Partition ID
            count: Number of indexes needed

        Returns:
            list: List of allocated global_indexes
        """
        if count <= 0:
            raise ValueError(f"Number of indexes needed must larger than 0, but got {count}")
        indexes = []

        # Get indexes from reusable pool
        if self.reusable_indexes:
            # Calculate number of indexes needed from reusable pool
            num_reuse = min(count, len(self.reusable_indexes))

            # Use slice operation to get multiple elements at once (FIFO principle)
            indexes.extend(self.reusable_indexes[:num_reuse])
            del self.reusable_indexes[:num_reuse]

        # If reusable pool doesn't have enough indexes, allocate new ones
        if len(indexes) < count:
            # Ensure newly allocated indexes don't conflict with existing ones
            needed = count - len(indexes)
            # Batch allocate consecutive index ranges
            start_index = self.global_index_counter
            end_index = start_index + needed

            # Directly generate consecutive index list
            new_indexes = list(range(start_index, end_index))

            # Batch update status
            self.allocated_indexes.update(new_indexes)
            self.global_index_counter = end_index

            indexes.extend(new_indexes)

        # Record partition-index relationship
        self.partition_to_indexes[partition_id].update(indexes)

        return indexes

    def release_indexes(self, partition_id):
        """
        Release all global_indexes of the specified partition, adding them to reusable pool.

        Args:
            partition_id: Partition ID

        Returns:
            list: List of released global_indexes
        """
        if partition_id in self.partition_to_indexes:
            indexes = self.partition_to_indexes.pop(partition_id)

            # Add released indexes to reusable pool
            self.reusable_indexes.extend(indexes)

            # Remove these indexes from allocated_indexes
            for idx in indexes:
                self.allocated_indexes.discard(idx)

            return indexes
        return []

    def get_indexes_for_partition(self, partition_id):
        """
        Get all global_indexes for the specified partition.

        Args:
            partition_id: Partition ID

        Returns:
            set: Set of global_indexes for this partition
        """
        return self.partition_to_indexes.get(partition_id, set()).copy()


@dataclass
class DataPartitionStatus:
    """
    Robust status information for a data partition with dynamic expansion support.

    This class tracks the production and consumption status of data within a specific
    partition (e.g., "train@global_batch_0", "inference@kv_cache_1") with full support
    for dynamic row and column expansion.
    """

    partition_id: str
    created_at: float = field(default_factory=time.time)

    # Production status tensor - dynamically expandable
    # Values: 0 = not produced, 1 = ready for consumption
    production_status: Optional[torch.Tensor] = torch.zeros(TQ_INIT_SAMPLE_NUM, TQ_INIT_FIELD_NUM, dtype=torch.int8)

    # Consumption status per task - task_name -> consumption_tensor
    # Each tensor tracks which samples have been consumed by that task
    consumption_status: dict[str, torch.Tensor] = field(default_factory=dict)

    # Field metadata
    field_name_mapping: dict[str, int] = field(default_factory=dict)  # field_name -> column_index
    field_dtypes: dict[int, dict[str, Any]] = field(default_factory=dict)  # global_idx -> {field: dtype}
    field_shapes: dict[int, dict[str, Any]] = field(default_factory=dict)  # global_idx -> {field: shape}

    # Dynamic configuration - these are computed from the current state
    @property
    def total_samples_num(self) -> int:
        """Current number of samples (rows) in the partition."""
        return self.production_status.shape[0] if self.production_status is not None else 0

    @property
    def total_fields_num(self) -> int:
        """Current number of fields (columns) in the partition."""
        return len(self.field_name_mapping)

    @property
    def allocated_fields_num(self) -> int:
        """Current number of allocated columns in the tensor."""
        return self.production_status.shape[1] if self.production_status is not None else 0

    @property
    def allocated_samples_num(self) -> int:
        """Current number of allocated rows in the tensor."""
        return self.production_status.shape[0] if self.production_status is not None else 0

    # ==================== Dynamic Expansion Methods ====================

    def ensure_samples_capacity(self, required_samples: int) -> bool:
        """
        Ensure the production status tensor has enough rows for the required samples.
        Dynamically expands if needed using unified minimum expansion size.

        Args:
            required_samples: Minimum number of samples needed
        """
        current_samples = self.production_status.shape[0]
        if required_samples > current_samples:
            # Expand rows using minimum expansion size for predictable memory usage
            expansion_needed = required_samples - current_samples
            min_expansion = max(TQ_SAMPLE_MIN_EXPANSION_SIZE, expansion_needed)
            new_samples = current_samples + min_expansion
            new_fields = self.production_status.shape[1]

            expanded_tensor = torch.zeros(new_samples, new_fields, dtype=torch.int8)
            expanded_tensor[:current_samples, :] = self.production_status
            self.production_status = expanded_tensor

            # Update consumption tensors for all tasks
            for task_name, consumption_tensor in self.consumption_status.items():
                expanded_consumption = torch.zeros(new_samples, dtype=torch.int8)
                expanded_consumption[:current_samples] = consumption_tensor
                self.consumption_status[task_name] = expanded_consumption

            logger.debug(
                f"Expanded partition {self.partition_id} from {current_samples} to {new_samples} samples "
                f"(added {min_expansion} samples)"
            )

    def ensure_fields_capacity(self, required_fields: int) -> bool:
        """
        Ensure the production status tensor has enough columns for the required fields.
        Dynamically expands if needed using unified minimum expansion size.

        Args:
            required_fields: Minimum number of fields needed
        """
        if self.production_status is None:
            # Will be initialized when samples are added
            return

        current_fields = self.production_status.shape[1]
        if required_fields > current_fields:
            # Expand columns using minimum expansion size for predictable memory usage
            expansion_needed = required_fields - current_fields
            min_expansion = max(TQ_FIELD_MIN_EXPANSION_SIZE, expansion_needed)
            new_fields = current_fields + min_expansion
            new_samples = self.production_status.shape[0]

            expanded_tensor = torch.zeros(new_samples, new_fields, dtype=torch.int8)
            expanded_tensor[:, :current_fields] = self.production_status
            self.production_status = expanded_tensor

            logger.debug(
                f"Expanded partition {self.partition_id} from {current_fields} to {new_fields} fields "
                f"(added {min_expansion} fields)"
            )

    # ==================== Production Status Interface ====================

    def update_production_status(
        self,
        global_indices: list[int],
        field_names: list[str],
        dtypes: Optional[dict[int, dict[str, Any]]] = None,
        shapes: Optional[dict[int, dict[str, Any]]] = None,
    ) -> bool:
        """
        Update production status for specific samples and fields.
        Handles dynamic expansion of both samples and fields.

        Args:
            global_indices: List of sample indices to update
            field_names: List of field names to mark as produced
            dtypes: Optional per-sample field dtype information
            shapes: Optional per-sample field shape information

        Returns:
            True if update was successful, False on error
        """
        try:
            # Determine required capacity
            max_sample_idx = max(global_indices) if global_indices else -1
            required_samples = max_sample_idx + 1

            # Ensure we have enough rows
            self.ensure_samples_capacity(required_samples)

            # Register new fields if needed
            new_fields = [field for field in field_names if field not in self.field_name_mapping]
            if new_fields:
                # Add new fields to mapping
                for field in new_fields:
                    self.field_name_mapping[field] = len(self.field_name_mapping)

                required_fields = len(self.field_name_mapping)
                self.ensure_fields_capacity(required_fields)

            # Update production status
            if self.production_status is not None and global_indices and field_names:
                field_indices = [self.field_name_mapping.get(field) for field in field_names]
                self.production_status[torch.tensor(global_indices)[:, None], torch.tensor(field_indices)] = 1

            # Update field metadata
            self._update_field_metadata(global_indices, field_names, dtypes, shapes)

            return True

        except Exception as e:
            logger.error(f"Error updating production status for partition {self.partition_id}: {e}")
            return False

    def _update_field_metadata(
        self,
        global_indices: list[int],
        field_names: list[str],
        dtypes: Optional[dict[int, dict[str, Any]]] = None,
        shapes: Optional[dict[int, dict[str, Any]]] = None,
    ):
        """Update field dtype and shape metadata."""
        for global_idx in global_indices:
            if global_idx not in self.field_dtypes:
                self.field_dtypes[global_idx] = {}
            if global_idx not in self.field_shapes:
                self.field_shapes[global_idx] = {}

            for field_name in field_names:
                if dtypes and global_idx in dtypes and field_name in dtypes[global_idx]:
                    self.field_dtypes[global_idx][field_name] = dtypes[global_idx][field_name]
                if shapes and global_idx in shapes and field_name in shapes[global_idx]:
                    self.field_shapes[global_idx][field_name] = shapes[global_idx][field_name]

    # ==================== Consumption Status Interface ====================

    def get_consumption_status(self, task_name: str) -> torch.Tensor:
        """
        Get or create consumption status for a specific task.
        Handles dynamic expansion when new samples are added.

        Args:
            task_name: Name of the consumer task

        Returns:
            Consumption status tensor for the specified task
        """
        if task_name not in self.consumption_status:
            if self.production_status is not None:
                self.consumption_status[task_name] = torch.zeros(self.total_samples_num, dtype=torch.int8)
            else:
                self.consumption_status[task_name] = torch.zeros(0, dtype=torch.int8)

        return self.consumption_status[task_name]

    # TODO: No need return, just raise error. Same With other function
    def mark_consumed(self, task_name: str, global_indices: list[int]) -> bool:
        """
        Mark specific samples as consumed by a task.

        Args:
            task_name: Name of the consumer task
            global_indices: List of sample indices to mark as consumed

        Returns:
            True if successful, False on error
        """
        try:
            consumption_status = self.get_consumption_status(task_name)
            if consumption_status.numel() > 0 and global_indices:
                consumption_status[global_indices] = 1
            return True
        except Exception as e:
            logger.error(f"Error marking samples consumed for partition {self.partition_id}, task {task_name}: {e}")
            return False

    # ==================== Data Scanning and Query Methods ====================

    def scan_data_status(self, field_names: list[str], task_name: str) -> list[int]:
        """
        Scan data status to find samples ready for consumption.
        This replaces the original _scan_data_status functionality.

        Args:
            field_names: List of required field names
            task_name: Name of the consumer task

        Returns:
            List of sample indices that are ready for consumption
        """
        if self.production_status is None:
            return []

        # Check if all requested fields are registered
        for field_name in field_names:
            if field_name not in self.field_name_mapping:
                return []

        row_mask = torch.ones(self.total_samples_num, dtype=torch.bool)

        # Apply consumption filter (exclude already consumed samples)
        consumption_status = self.get_consumption_status(task_name)
        if consumption_status is not None:
            unconsumed_mask = consumption_status == 0
            row_mask &= unconsumed_mask

        # Create column mask for requested fields
        col_mask = torch.zeros(self.allocated_fields_num, dtype=torch.bool)
        field_indices = [self.field_name_mapping[field] for field in field_names]
        if field_indices:
            col_mask[field_indices] = True

        # Filter production status by masks
        relevant_status = self.production_status[row_mask][:, col_mask]

        # Check if all required fields are ready for each sample
        all_fields_ready = torch.all(relevant_status, dim=1)
        ready_indices_in_filtered = torch.nonzero(all_fields_ready, as_tuple=False).flatten()

        # Map back to original sample indices
        all_indices = torch.where(row_mask)[0]
        ready_sample_indices = all_indices[ready_indices_in_filtered].tolist()

        return ready_sample_indices

    # ==================== Field Metadata Methods ====================

    def get_field_dtype(self, global_index: int, field_name: str) -> Optional[Any]:
        """Get dtype for a specific sample and field."""
        return self.field_dtypes.get(global_index, {}).get(field_name)

    def get_field_shape(self, global_index: int, field_name: str) -> Optional[Any]:
        """Get shape for a specific sample and field."""
        return self.field_shapes.get(global_index, {}).get(field_name)

    # ==================== Statistics and Monitoring ====================

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics for this partition."""
        stats = {
            "partition_id": self.partition_id,
            "created_at": self.created_at,
            "total_samples_num": self.total_samples_num,
            "total_fields_num": self.total_fields_num,
            "allocated_fields_num": self.allocated_fields_num,
            "registered_tasks": list(self.consumption_status.keys()),
        }

        if self.production_status is not None:
            produced_samples = torch.any(self.production_status == 1, dim=1).sum().item()
            stats["produced_samples"] = produced_samples
            stats["production_progress"] = (
                produced_samples / self.total_samples_num if self.total_samples_num > 0 else 0
            )

            # Field-wise production statistics
            field_stats = {}
            for field_name, field_idx in self.field_name_mapping.items():
                field_produced = (self.production_status[:, field_idx] == 1).sum().item()
                field_stats[field_name] = {
                    "produced_samples": field_produced,
                    "production_progress": field_produced / self.total_samples_num if self.total_samples_num > 0 else 0,
                }
            stats["field_statistics"] = field_stats

        # Consumption statistics per task
        consumption_stats = {}
        for task_name, consumption_tensor in self.consumption_status.items():
            consumed_samples = (consumption_tensor == 1).sum().item()
            consumption_stats[task_name] = {
                "consumed_samples": consumed_samples,
                "consumption_progress": consumed_samples / self.total_samples_num if self.total_samples_num > 0 else 0,
            }
        stats["consumption_statistics"] = consumption_stats

        return stats

    def clear_data(self, global_indexes_range: list[int], clear_consumption: bool = True) -> bool:
        """Clear all production and optionally consumption data."""
        try:
            if self.production_status is not None:
                self.production_status[global_indexes_range, :] = 0

            if clear_consumption:
                for consumption_tensor in self.consumption_status.values():
                    consumption_tensor[global_indexes_range] = 0

            return True
        except Exception as e:
            logger.error(f"Error clearing data for partition {self.partition_id}: {e}")
            return False


@ray.remote(num_cpus=1)
class TransferQueueController:
    """
    TransferQueue Controller with partition-based data management.

    This refactored controller manages data through dynamic partitions instead of
    fixed global batches. Each partition represents a logical data container
    (e.g., "train@global_batch_0", "inference@kv_cache_1") that can be created
    on-demand and managed independently.

    Key improvements:
    - Dynamic partition creation on-demand
    - No dependency on training-specific parameters (global_batch_size, etc.)
    - Support for diverse use cases (KV cache migration, model resharding, etc.)
    - Flexible data organization through partition-based addressing
    """

    def __init__(self, sampler: BaseSampler | type[BaseSampler] = SequentialSampler) -> None:
        """Initialize the TransferQueue Controller.

        Args:
            sampler: Sampler instance or sampler class to use for data sampling.
                    - If a BaseSampler instance is provided, it will be used directly
                    - If a BaseSampler subclass is provided, it will be instantiated
                    - Defaults to SequentialSampler for simple sequential sampling
                    - Example: sampler=GRPOGroupNSampler() (instance)
                    - Example: sampler=GRPOGroupNSampler (class)
        """
        if isinstance(sampler, BaseSampler):
            self.sampler = sampler
        elif isinstance(sampler, type) and issubclass(sampler, BaseSampler):
            self.sampler = sampler()
        else:
            raise TypeError(
                f"sampler {getattr(sampler, '__name__', repr(sampler))} must be an instance or subclass of BaseSampler"
            )

        self.controller_id = f"TQ_CONTROLLER_{uuid4().hex[:8]}"

        # Initialize ZMQ sockets for communication
        self._init_zmq_socket()

        # Partition management
        self.partitions: dict[str, DataPartitionStatus] = {}  # partition_id -> DataPartitionStatus

        # Partition-GlobalIndex management
        self.index_manager = PartitionIndexManager()  # partition_id -> global_indexes

        # Connected storage managers tracking
        self._connected_storage_managers: set[str] = set()

        # Start background processing threads
        self._start_process_handshake()
        self._start_process_update_data_status()
        self._start_process_request()

        logger.info(f"TransferQueue Controller {self.controller_id} initialized")

    # ==================== Partition Management API ====================

    def create_partition(self, partition_id: str) -> bool:
        """
        Create a new data partition.

        Note: Partitions now dynamically expand as needed, so initial capacity is not required.

        Args:
            partition_id: Unique identifier for the partition (e.g., "train@global_batch_0")

        Returns:
            True if partition was created successfully, False if it already exists
        """
        if partition_id in self.partitions:
            logger.warning(f"Partition {partition_id} already exists")
            return False

        self.partitions[partition_id] = DataPartitionStatus(partition_id=partition_id)

        logger.info(f"Created partition {partition_id} with dynamic capacity")
        return True

    def get_partition(self, partition_id: str) -> Optional[DataPartitionStatus]:
        """
        Get partition status information.

        Args:
            partition_id: ID of the partition to retrieve

        Returns:
            DataPartitionStatus object if partition exists, None otherwise
        """
        return self.partitions.get(partition_id)

    def list_partitions(self) -> list[str]:
        """
        List all available partition IDs.

        Returns:
            List of partition IDs
        """
        return list(self.partitions.keys())

    def delete_partition(self, partition_id: str) -> bool:
        """
        Delete a partition and all its data.

        Args:
            partition_id: ID of the partition to delete

        Returns:
            True if partition was deleted, False if it didn't exist
        """
        if partition_id in self.partitions:
            del self.partitions[partition_id]
            logger.info(f"Deleted partition {partition_id}")
            return True
        return False

    # ==================== Partition Index Management API ====================

    def get_partition_index_range(self, partition: DataPartitionStatus) -> set:
        """
        Get all indexes for a specific partition.

        Args:
            partition: Partition identifier

        Returns:
            Set of indexes allocated to the partition
        """
        return self.index_manager.get_indexes_for_partition(partition)

    # ==================== Data Production API ====================

    # TODO: Modify dtypes & shapes to be required
    def update_production_status(
        self,
        partition_id: str,
        global_indexes: list[int],
        field_names: list[str],
        dtypes: Optional[dict[int, dict[str, Any]]] = None,
        shapes: Optional[dict[int, dict[str, Any]]] = None,
    ) -> bool:
        """
        Update production status for specific samples and fields in a partition.
        Delegates to the partition's own update_production_status method.

        Args:
            partition_id: ID of the partition
            global_indexes: List of sample indices to update
            field_names: List of field names to mark as produced
            dtypes: Optional per-sample field dtype information
            shapes: Optional per-sample field shape information

        Returns:
            True if update was successful, False otherwise
        """
        partition = self.get_partition(partition_id)
        if not partition:
            logger.error(f"Partition {partition_id} not found")
            return False

        success = partition.update_production_status(global_indexes, field_names, dtypes, shapes)
        if success:
            logger.debug(
                f"Updated production status for partition {partition_id}: samples={global_indexes}, "
                f"fields={field_names}"
            )
        return success

    # ==================== Data Consumption API ====================

    def get_consumption_status(self, partition_id: str, task_name: str) -> Optional[torch.Tensor]:
        """
        Get or create consumption status for a specific task and partition.
        Delegates to the partition's own method.

        Args:
            partition_id: ID of the partition
            task_name: Name of the consumer task

        Returns:
            Consumption status tensor if partition exists, None otherwise
        """
        partition = self.get_partition(partition_id)
        if not partition:
            return None

        return partition.get_consumption_status(task_name)

    def get_metadata(
        self,
        data_fields: list[str],
        partition_id: str,
        mode: str = "fetch",
        task_name: str | None = None,
        batch_size: int | None = None,
        sampling_config: Optional[dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> BatchMeta:
        """
        Retrieve metadata with support for three modes.

        Args:
            data_fields: List of field names to include in metadata
            partition_id: Partition id for which to retrieve metadata
            mode: Operation mode - 'insert', 'fetch', or 'force_fetch'
                - mode="insert": Create metadata for new samples (for data insertion)
                - mode="fetch": Get metadata from ready samples using the configured sampler
                - mode="force_fetch": Get metadata for unconsumed samples without sampling
                                      (excludes already consumed samples)
            task_name: Name of the consumer task (required for fetch modes)
            batch_size: Number of samples to retrieve
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            BatchMeta object containing the requested metadata

        Raises:
            TimeoutError: If waiting for sufficient data times out in fetch mode
        """
        if partition_id not in self.partitions:
            self.create_partition(partition_id)

        if mode == "insert":
            if data_fields:
                # First put_data call, get_metadata in insert mode
                batch_global_indexes = self.index_manager.allocate_indexes(partition_id, count=batch_size)
            else:
                # clear metadata call passes empty data_fields
                batch_global_indexes = self.index_manager.get_indexes_for_partition(partition_id)
            return self.generate_batch_meta(partition_id, batch_global_indexes, data_fields, mode)

        assert task_name is not None
        if mode == "fetch":
            # Find ready samples within current data partition and package into BatchMeta when reading

            start_time = time.time()
            while True:
                # ready_for_consume_indexes: samples where all required fields are produced
                # (production status is ready) and not yet consumed
                ready_for_consume_indexes = self.scan_data_status(partition_id, data_fields, task_name, batch_size)

                if len(ready_for_consume_indexes) < batch_size:
                    if time.time() - start_time > TQ_CONTROLLER_GET_METADATA_TIMEOUT:
                        raise TimeoutError(
                            f"Timeout while waiting for sufficient data. "
                            f"Required: {batch_size}, Available: {len(ready_for_consume_indexes)}"
                        )
                    logger.warning(
                        f"Insufficient complete groups available. Required: {batch_size}, "
                        f"Available: {len(ready_for_consume_indexes)}. Retrying in "
                        f"{TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                    )
                    time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
                    continue

                # Try sampling - if it returns empty lists, retry
                batch_global_indexes, consumed_indexes = self.sampler(
                    ready_for_consume_indexes,
                    batch_size,
                    **(sampling_config or {}),
                )

                # Check if we got valid results from the sampler
                if len(batch_global_indexes) == batch_size:
                    break

                if time.time() - start_time > TQ_CONTROLLER_GET_METADATA_TIMEOUT:
                    raise TimeoutError(
                        f"Timeout while waiting for sufficient data. "
                        f"Required: {batch_size}, Available: {len(ready_for_consume_indexes)}, "
                        f"Sampled: {len(batch_global_indexes)}"
                    )

                logger.warning(
                    f"Insufficient complete groups available. Required: {batch_size}, "
                    f"Available: {len(ready_for_consume_indexes)}, "
                    f"Sampled: {len(batch_global_indexes)}. Retrying in "
                    f"{TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                )
                time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
            logger.debug(f"ready for consume idx: {ready_for_consume_indexes}")
            logger.debug(f"sampled idx: {batch_global_indexes}")
        elif mode == "force_fetch":
            global_indexes_range = self.index_manager.get_indexes_for_partition(partition_id)
            consumer_status = self.get_consumption_status(partition_id, task_name)
            not_consumed_idx = [i for i in global_indexes_range if consumer_status[i] == 0]
            batch_global_indexes = not_consumed_idx
            consumed_indexes = []

        # Package into metadata
        metadata = self.generate_batch_meta(partition_id, batch_global_indexes, data_fields, mode)

        # Mark samples as consumed if in fetch mode
        if mode == "fetch" and consumed_indexes:
            partition = self.partitions[partition_id]
            partition.mark_consumed(task_name, consumed_indexes)

        logger.debug(f"get_metadata: {metadata}")

        return metadata

    def scan_data_status(
        self,
        partition_id: str,
        data_fields: list[str],
        task_name: str,
        batch_size: int,
        sample_filter: Optional[list[int]] = None,
        timeout: float = TQ_CONTROLLER_GET_METADATA_TIMEOUT,
    ) -> list[int]:
        """
        Find samples that are ready for consumption in a specific partition.
        Delegates scanning functionality to the partition's own method.

        Args:
            partition_id: ID of the partition
            data_fields: List of required field names
            task_name: Name of the consumer task
            batch_size: Number of samples needed
            sample_filter: Optional list of specific sample indices to consider
            timeout: Maximum time to wait for sufficient data

        Returns:
            List of sample indices that are ready for consumption

        Raises:
            TimeoutError: If sufficient data is not available within timeout
        """
        start_time = time.time()

        while True:
            partition = self.get_partition(partition_id)
            if not partition:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Partition {partition_id} not found")
                time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
                continue

            # Use partition's own scanning method
            ready_sample_indices = partition.scan_data_status(data_fields, task_name)

            if len(ready_sample_indices) >= batch_size:
                return ready_sample_indices[:batch_size]

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timeout waiting for sufficient data in partition {partition_id}. "
                    f"Required: {batch_size}, Available: {len(ready_sample_indices)}"
                )

            logger.warning(
                f"Insufficient data in partition {partition_id}. Required: {batch_size}, "
                f"Available: {len(ready_sample_indices)}. Retrying in {TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
            )
            time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)

    # ==================== Metadata Generation API ====================

    def generate_batch_meta(
        self,
        partition_id: str,
        batch_global_indexes: list[int],
        data_fields: list[str],
        mode: str = "fetch",
    ) -> BatchMeta:
        """
        Generate BatchMeta for specific samples in a partition.

        This function is responsible only for metadata generation and does not
        modify consumption state. State management is handled by the calling function.

        Args:
            partition_id: ID of the partition
            batch_global_indexes: List of sample indices to include in the batch
            data_fields: List of field names to include
            mode: Operation mode - 'fetch', 'insert', or 'force_fetch'

        Returns:
            BatchMeta object containing sample metadata

        Raises:
            ValueError: If partition doesn't exist or invalid mode
        """
        partition = self.get_partition(partition_id)
        if not partition:
            raise ValueError(f"Partition {partition_id} not found")

        if mode not in ["fetch", "insert", "force_fetch"]:
            raise ValueError(f"Invalid mode: {mode}")

        # Generate sample metadata
        samples = []
        for global_index in batch_global_indexes:
            fields = {}
            for field_name in data_fields:
                # Determine production status
                if mode == "fetch":
                    production_status = ProductionStatus.READY_FOR_CONSUME
                    dtype = partition.get_field_dtype(global_index, field_name)
                    shape = partition.get_field_shape(global_index, field_name)
                elif mode == "insert":
                    production_status = ProductionStatus.NOT_PRODUCED
                    dtype = None
                    shape = None
                elif mode == "force_fetch":
                    field_index = partition.field_name_mapping.get(field_name)
                    if (
                        field_index is not None
                        and partition.production_status is not None
                        and partition.production_status[global_index, field_index] == 1
                    ):
                        production_status = ProductionStatus.NOT_PRODUCED
                        dtype = partition.get_field_dtype(global_index, field_name)
                        shape = partition.get_field_shape(global_index, field_name)
                    else:
                        production_status = ProductionStatus.NOT_PRODUCED
                        dtype = None
                        shape = None

                fields[field_name] = FieldMeta(
                    name=field_name,
                    dtype=dtype,
                    shape=shape,
                    production_status=production_status,
                )

            sample = SampleMeta(
                partition_id=partition_id,
                global_index=global_index,
                fields=fields,
            )
            samples.append(sample)

        return BatchMeta(samples=samples)

    def clear(self, partition_id: str, clear_consumption: bool = True) -> bool:
        """
        Clear data for a specific partition.

        Args:
            partition_id: ID of the partition to clear
            clear_consumption: Whether to also clear consumption status

        Returns:
            True if cleared successfully, False otherwise
        """
        partition = self.get_partition(partition_id)
        if not partition:
            raise ValueError(f"Partition {partition_id} not found")

        global_indexes_range = list(self.index_manager.get_indexes_for_partition(partition_id))
        success = partition.clear_data(global_indexes_range, clear_consumption)
        self.index_manager.release_indexes(partition_id)
        if success:
            logger.info(f"Cleared data for partition {partition_id}")
        return success

    def _init_zmq_socket(self):
        """Initialize ZMQ sockets for communication."""
        self.zmq_context = zmq.Context()
        self._node_ip = get_node_ip_address()
        self._handshake_socket_port = get_free_port()
        self._request_handle_socket_port = get_free_port()
        self._data_status_update_socket_port = get_free_port()

        self.handshake_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.handshake_socket.bind(f"tcp://{self._node_ip}:{self._handshake_socket_port}")

        self.request_handle_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.request_handle_socket.bind(f"tcp://{self._node_ip}:{self._request_handle_socket_port}")

        self.data_status_update_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.data_status_update_socket.bind(f"tcp://{self._node_ip}:{self._data_status_update_socket_port}")

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.CONTROLLER,
            id=self.controller_id,
            ip=self._node_ip,
            ports={
                "handshake_socket": self._handshake_socket_port,
                "request_handle_socket": self._request_handle_socket_port,
                "data_status_update_socket": self._data_status_update_socket_port,
            },
        )

    def _wait_connection(self):
        """Wait for storage instances to complete handshake with retransmission support."""
        poller = zmq.Poller()
        poller.register(self.handshake_socket, zmq.POLLIN)

        logger.info(f"Dynamic Controller {self.controller_id} started waiting for storage connections...")

        while True:
            socks = dict(poller.poll(TQ_CONTROLLER_CONNECTION_CHECK_INTERVAL * 1000))

            if self.handshake_socket in socks:
                try:
                    identity, serialized_msg = self.handshake_socket.recv_multipart()
                    request_msg = ZMQMessage.deserialize(serialized_msg)

                    if request_msg.request_type == ZMQRequestType.HANDSHAKE:
                        storage_manager_id = request_msg.sender_id

                        # Always send ACK for HANDSHAKE
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.HANDSHAKE_ACK,
                            sender_id=self.controller_id,
                            body={},
                        ).serialize()
                        self.handshake_socket.send_multipart([identity, response_msg])

                        # Track new connections
                        if storage_manager_id not in self._connected_storage_managers:
                            self._connected_storage_managers.add(storage_manager_id)
                            storage_manager_type = request_msg.body.get("storage_manager_type", "Unknown")
                            logger.info(
                                f"Dynamic Controller {self.controller_id} received handshake from "
                                f"storage manager {storage_manager_id} (type: {storage_manager_type}). "
                                f"Total connected: {len(self._connected_storage_managers)}"
                            )
                        else:
                            logger.debug(
                                f"Dynamic Controller {self.controller_id} received duplicate handshake from "
                                f"storage manager {storage_manager_id}. Resending ACK."
                            )

                except Exception as e:
                    logger.error(f"Dynamic Controller {self.controller_id} error processing handshake: {e}")

    def _start_process_handshake(self):
        """Start the handshake process thread."""
        self.wait_connection_thread = Thread(
            target=self._wait_connection, name="DynamicTransferQueueControllerWaitConnectionThread", daemon=True
        )
        self.wait_connection_thread.start()

    def _start_process_update_data_status(self):
        """Start the data status update processing thread."""
        self.process_update_data_status_thread = Thread(
            target=self._update_data_status,
            name="DynamicTransferQueueControllerProcessUpdateDataStatusThread",
            daemon=True,
        )
        self.process_update_data_status_thread.start()

    def _start_process_request(self):
        """Start the request processing thread."""
        self.process_request_thread = Thread(
            target=self._process_request, name="DynamicTransferQueueControllerProcessRequestThread", daemon=True
        )
        self.process_request_thread.start()

    def _process_request(self):
        """Main request processing loop - adapted for partition-based operations."""
        while True:
            identity, serialized_msg = self.request_handle_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)

            if request_msg.request_type == ZMQRequestType.GET_META:
                params = request_msg.body

                metadata = self.get_metadata(
                    data_fields=params["data_fields"],
                    batch_size=params["batch_size"],
                    partition_id=params["partition_id"],
                    mode=params.get("mode", "fetch"),
                    task_name=params.get("task_name"),
                    sampling_config=params.get("sampling_config"),
                )

                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )

            elif request_msg.request_type == ZMQRequestType.GET_CLEAR_META:
                params = request_msg.body
                partition_id = params["partition_id"]

                metadata = self.get_metadata(
                    data_fields=[],
                    partition_id=partition_id,
                    mode="insert",
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_CLEAR_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )
            elif request_msg.request_type == ZMQRequestType.CLEAR_META:
                params = request_msg.body
                partition_id = params["partition_id"]

                clear_success = self.clear(partition_id)
                if clear_success:
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.CLEAR_META_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"message": f"Clear operation completed by controller {self.controller_id}"},
                    )
                else:
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.CLEAR_META_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"error": f"Clear operation failed for partition {partition_id}"},
                    )

            elif request_msg.request_type == ZMQRequestType.CHECK_CONSUMPTION:
                # Handle consumption status checks
                params = request_msg.body

                consumption_status = self.get_consumption_status(params["partition_id"], params["task_name"])
                sample_filter = params.get("sample_filter")

                if consumption_status is not None and sample_filter:
                    batch_status = consumption_status[sample_filter]
                    consumed = torch.all(batch_status == 1).item()
                elif consumption_status is not None:
                    batch_status = consumption_status
                    consumed = torch.all(batch_status == 1).item()
                else:
                    consumed = False

                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={
                        "partition_id": params["partition_id"],
                        "consumed": consumed,
                    },
                )
            self.request_handle_socket.send_multipart([identity, response_msg.serialize()])

    def _update_data_status(self):
        """Process data status update messages from storage units - adapted for partitions."""
        while True:
            identity, serialized_msg = self.data_status_update_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)

            if request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE:
                message_data = request_msg.body
                partition_id = message_data.get("partition_id")

                # Update production status
                success = self.update_production_status(
                    partition_id=partition_id,
                    global_indexes=message_data.get("global_indexes", []),
                    field_names=message_data.get("fields", []),
                    dtypes=message_data.get("dtypes", {}),
                    shapes=message_data.get("shapes", {}),
                )

                if success:
                    logger.info(f"Updated production status for partition {partition_id}")

                # Send acknowledgment
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "partition_id": partition_id,
                        "success": success,
                    },
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get ZMQ server connection information."""
        return self.zmq_server_info
