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

TQ_CONTROLLER_GET_METADATA_TIMEOUT = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_TIMEOUT", 300))
TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL", 1))
TQ_CONTROLLER_CONNECTION_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_CONNECTION_CHECK_INTERVAL", 2))
TQ_INIT_FIELD_NUM = int(os.environ.get("TQ_INIT_FIELD_NUM", 10))

# Expansion configuration - Unified approach using minimum expansion sizes
TQ_SAMPLE_MIN_EXPANSION_SIZE = int(
    os.environ.get("TQ_SAMPLE_MIN_EXPANSION_SIZE", 10)
)  # Minimum expansion size for samples (rows)
TQ_FIELD_MIN_EXPANSION_SIZE = int(
    os.environ.get("TQ_FIELD_MIN_EXPANSION_SIZE", 5)
)  # Minimum expansion size for fields (columns)
TQ_INIT_SAMPLE_NUM = int(os.environ.get("TQ_INIT_SAMPLE_NUM", 10))  # Initial number of samples


class PartitionIndexManager:
    """
    管理分区与全局索引的映射关系，负责索引的分配和复用
    修复版本：解决了索引分配可能导致的覆盖问题
    """

    def __init__(self):
        # 记录每个分区使用的global_index集合
        self.partition_to_indexes = defaultdict(set)

        # 可复用的global_index池 - 使用列表存储
        self.reusable_indexes = []

        # 全局索引计数器，用于分配新的索引
        self.global_index_counter = 0

        # 跟踪所有已分配的索引（活跃的 + 可复用的）
        self.allocated_indexes = set()

    def allocate_indexes(self, partition_id, count=1):
        """
        为指定分区分配global_index
        优先从可复用池中获取，不足时分配新的索引

        修复点：
        1. 确保新分配的索引不会与现有活跃索引冲突
        2. 维护allocated_indexes集合跟踪所有已分配的索引
        3. 智能global_index_counter始终指向最大已分配索引+1

        Args:
            partition_id: 分区ID
            count: 需要分配的索引数量

        Returns:
            list: 分配的global_index列表
        """
        indexes = []

        # 从可复用池中获取索引
        if self.reusable_indexes and count > 0:
            # 计算需要从可复用池获取的索引数量
            num_reuse = min(count, len(self.reusable_indexes))

            # 使用切片操作一次性获取多个元素（从开头获取，FIFO原则）
            indexes.extend(self.reusable_indexes[:num_reuse])
            del self.reusable_indexes[:num_reuse]

            # 从allocated_indexes中移除这些索引（它们将重新变为活跃状态）
            for idx in indexes:
                self.allocated_indexes.discard(idx)

        # 如果可复用池中的索引不足，分配新的索引
        if len(indexes) < count:
            # 确保新分配的索引不会与现有索引冲突
            needed = count - len(indexes)
            new_indexes = []

            while len(new_indexes) < needed:
                # 检查当前计数器指向的索引是否已被使用
                if self.global_index_counter not in self.allocated_indexes:
                    new_indexes.append(self.global_index_counter)
                    self.allocated_indexes.add(self.global_index_counter)
                    self.global_index_counter += 1
                else:
                    # 如果已被使用，递增计数器直到找到可用的索引
                    self.global_index_counter += 1

            indexes.extend(new_indexes)

        # 记录分区与索引的关系
        self.partition_to_indexes[partition_id].update(indexes)

        return indexes

    def release_indexes(self, partition_id):
        """
        释放指定分区的所有global_index，将其加入可复用池

        修复点：
        1. 释放的索引添加到allocated_indexes集合
        2. 不修改global_index_counter，确保其始终指向最大已分配索引+1

        Args:
            partition_id: 分区ID

        Returns:
            list: 释放的global_index列表
        """
        if partition_id in self.partition_to_indexes:
            indexes = self.partition_to_indexes.pop(partition_id)

            # 将释放的索引添加到可复用池
            self.reusable_indexes.extend(indexes)

            # 将释放的索引添加到allocated_indexes集合
            self.allocated_indexes.update(indexes)

            return indexes
        return []

    def get_indexes_for_partition(self, partition_id):
        """
        获取指定分区的所有global_index

        Args:
            partition_id: 分区ID

        Returns:
            set: 该分区的global_index集合
        """
        return self.partition_to_indexes.get(partition_id, set()).copy()

    def get_allocated_indexes(self):
        """
        获取所有已分配的索引（活跃的 + 可复用的）

        Returns:
            set: 所有已分配的索引
        """
        # 活跃索引
        active_indexes = set()
        for indexes in self.partition_to_indexes.values():
            active_indexes.update(indexes)

        # 可复用索引
        reusable_indexes = set(self.reusable_indexes)

        # 返回所有已分配的索引
        return active_indexes.union(reusable_indexes)


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
    production_status: Optional[torch.Tensor] = None

    # Consumption status per task - task_name -> consumption_tensor
    # Each tensor tracks which samples have been consumed by that task
    consumption_status: dict[str, torch.Tensor] = field(default_factory=dict)

    # Field metadata
    field_name_mapping: dict[str, int] = field(default_factory=dict)  # field_name -> column_index
    field_dtypes: dict[int, dict[str, Any]] = field(default_factory=dict)  # sample_idx -> {field: dtype}
    field_shapes: dict[int, dict[str, Any]] = field(default_factory=dict)  # sample_idx -> {field: shape}

    # Dynamic configuration - these are computed from the current state
    @property
    def total_samples(self) -> int:
        """Current number of samples (rows) in the partition."""
        return self.production_status.shape[0] if self.production_status is not None else 0

    @property
    def total_fields(self) -> int:
        """Current number of fields (columns) in the partition."""
        return len(self.field_name_mapping)

    @property
    def allocated_fields(self) -> int:
        """Current number of allocated columns in the tensor."""
        return self.production_status.shape[1] if self.production_status is not None else 0

    # ==================== Dynamic Expansion Methods ====================

    def ensure_samples_capacity(self, required_samples: int) -> bool:
        """
        Ensure the production status tensor has enough rows for the required samples.
        Dynamically expands if needed using unified minimum expansion size.

        Args:
            required_samples: Minimum number of samples needed

        Returns:
            True if expansion was successful or not needed, False on error
        """
        if self.production_status is None:
            # First-time initialization - use configured initial size
            initial_size = max(TQ_INIT_SAMPLE_NUM, required_samples)
            self.production_status = torch.zeros(initial_size, TQ_INIT_FIELD_NUM, dtype=torch.int8)
            logger.debug(
                f"Initialized production status for partition {self.partition_id}: "
                f"{initial_size} samples, {TQ_INIT_FIELD_NUM} fields"
            )
            return True

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
            return True

        return True

    def ensure_fields_capacity(self, required_fields: int) -> bool:
        """
        Ensure the production status tensor has enough columns for the required fields.
        Dynamically expands if needed using unified minimum expansion size.

        Args:
            required_fields: Minimum number of fields needed

        Returns:
            True if expansion was successful or not needed, False on error
        """
        if self.production_status is None:
            # Will be initialized when samples are added
            return True

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
            return True

        return True

    # ==================== Production Status Interface ====================

    def update_production_status(
        self,
        sample_indices: list[int],
        field_names: list[str],
        dtypes: Optional[dict[int, dict[str, Any]]] = None,
        shapes: Optional[dict[int, dict[str, Any]]] = None,
    ) -> bool:
        """
        Update production status for specific samples and fields.
        Handles dynamic expansion of both samples and fields.

        Args:
            sample_indices: List of sample indices to update
            field_names: List of field names to mark as produced
            dtypes: Optional per-sample field dtype information
            shapes: Optional per-sample field shape information

        Returns:
            True if update was successful, False on error
        """
        self.controller_id = f"DYNAMIC_TQ_CONTROLLER_{uuid4().hex[:8]}"
        try:
            # Determine required capacity
            num_samples = len(sample_indices) if sample_indices else -1
            # TODO: 确认为什么这里要+1
            # required_samples = num_samples + 1
            required_samples = num_samples

            # Register new fields if needed
            new_fields = [field for field in field_names if field not in self.field_name_mapping]
            if new_fields:
                # Add new fields to mapping
                for field in new_fields:
                    self.field_name_mapping[field] = len(self.field_name_mapping)

                required_fields = len(self.field_name_mapping)
                self.ensure_fields_capacity(required_fields)

            # Ensure we have enough rows
            self.ensure_samples_capacity(required_samples)

            # Update production status
            if self.production_status is not None and sample_indices and field_names:
                field_indices = [self.field_name_mapping.get(field) for field in field_names]
                self.production_status[torch.tensor(sample_indices)[:, None], torch.tensor(field_indices)] = 1

            # Update field metadata
            self._update_field_metadata(sample_indices, field_names, dtypes, shapes)

            return True

        except Exception as e:
            logger.error(f"Error updating production status for partition {self.partition_id}: {e}")
            return False

    def _update_field_metadata(
        self,
        sample_indices: list[int],
        field_names: list[str],
        dtypes: Optional[dict[int, dict[str, Any]]] = None,
        shapes: Optional[dict[int, dict[str, Any]]] = None,
    ):
        """Update field dtype and shape metadata."""
        for sample_idx in sample_indices:
            if sample_idx not in self.field_dtypes:
                self.field_dtypes[sample_idx] = {}
            if sample_idx not in self.field_shapes:
                self.field_shapes[sample_idx] = {}

            for field_name in field_names:
                if dtypes and sample_idx in dtypes and field_name in dtypes[sample_idx]:
                    self.field_dtypes[sample_idx][field_name] = dtypes[sample_idx][field_name]
                if shapes and sample_idx in shapes and field_name in shapes[sample_idx]:
                    self.field_shapes[sample_idx][field_name] = shapes[sample_idx][field_name]

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
                self.consumption_status[task_name] = torch.zeros(self.total_samples, dtype=torch.int8)
            else:
                self.consumption_status[task_name] = torch.zeros(0, dtype=torch.int8)

        # Ensure consumption tensor has same number of rows as production tensor
        consumption_tensor = self.consumption_status[task_name]
        if self.production_status is not None and consumption_tensor.shape[0] < self.total_samples:
            expanded_consumption = torch.zeros(self.total_samples, dtype=torch.int8)
            expanded_consumption[: consumption_tensor.shape[0]] = consumption_tensor
            self.consumption_status[task_name] = expanded_consumption

        return self.consumption_status[task_name]

    def mark_consumed(self, task_name: str, sample_indices: list[int]) -> bool:
        """
        Mark specific samples as consumed by a task.

        Args:
            task_name: Name of the consumer task
            sample_indices: List of sample indices to mark as consumed

        Returns:
            True if successful, False on error
        """
        try:
            consumption_status = self.get_consumption_status(task_name)
            if consumption_status is not None and sample_indices:
                consumption_status[sample_indices] = 1
            return True
        except Exception as e:
            logger.error(f"Error marking samples consumed for partition {self.partition_id}, task {task_name}: {e}")
            return False

    # ==================== Data Scanning and Query Methods ====================

    def scan_data_status(
        self, field_names: list[str], task_name: str, sample_filter: Optional[list[int]] = None
    ) -> list[int]:
        """
        Scan data status to find samples ready for consumption.
        This replaces the original _scan_data_status functionality.

        Args:
            field_names: List of required field names
            task_name: Name of the consumer task
            sample_filter: Optional list of specific sample indices to consider

        Returns:
            List of sample indices that are ready for consumption
        """
        if self.production_status is None:
            return []

        # Check if all requested fields are registered
        for field in field_names:
            if field not in self.field_name_mapping:
                return []

        # Create row mask
        if sample_filter is not None:
            row_mask = torch.zeros(self.total_samples, dtype=torch.bool)
            valid_indices = [idx for idx in sample_filter if idx < self.total_samples]
            if valid_indices:
                row_mask[valid_indices] = True
        else:
            row_mask = torch.ones(self.total_samples, dtype=torch.bool)

        # Apply consumption filter (exclude already consumed samples)
        consumption_status = self.get_consumption_status(task_name)
        if consumption_status is not None:
            unconsumed_mask = consumption_status == 0
            row_mask &= unconsumed_mask

        # Create column mask for requested fields
        col_mask = torch.zeros(self.allocated_fields, dtype=torch.bool)
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

    def generate_data_status_mask(
        self, field_names: list[str], task_name: str, sample_filter: Optional[list[int]] = None
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate data availability mask for this partition.
        This replaces the original generate_data_status_mask functionality.

        Args:
            field_names: List of field names to check
            task_name: Name of the consumer task
            sample_filter: Optional list of specific sample indices to consider

        Returns:
            Tuple of (row_mask, col_mask) tensors, or (None, None) if not available
        """
        if self.production_status is None:
            return None, None

        # Check if all requested fields are registered
        for field in field_names:
            if field not in self.field_name_mapping:
                return None, None

        # Create row mask
        if sample_filter is not None:
            row_mask = torch.zeros(self.total_samples, dtype=torch.bool)
            valid_indices = [idx for idx in sample_filter if idx < self.total_samples]
            if valid_indices:
                row_mask[valid_indices] = True
        else:
            row_mask = torch.ones(self.total_samples, dtype=torch.bool)

        # Apply consumption filter
        consumption_status = self.get_consumption_status(task_name)
        if consumption_status is not None:
            unconsumed_mask = consumption_status == 0
            row_mask &= unconsumed_mask

        # Create column mask for requested fields
        col_mask = torch.zeros(self.allocated_fields, dtype=torch.bool)
        field_indices = [self.field_name_mapping[field] for field in field_names]
        if field_indices:
            col_mask[field_indices] = True

        return row_mask, col_mask

    # ==================== Field Metadata Methods ====================

    def get_field_dtype(self, sample_idx: int, field_name: str) -> Optional[Any]:
        """Get dtype for a specific sample and field."""
        return self.field_dtypes.get(sample_idx, {}).get(field_name)

    def get_field_shape(self, sample_idx: int, field_name: str) -> Optional[Any]:
        """Get shape for a specific sample and field."""
        return self.field_shapes.get(sample_idx, {}).get(field_name)

    # ==================== Statistics and Monitoring ====================

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics for this partition."""
        stats = {
            "partition_id": self.partition_id,
            "created_at": self.created_at,
            "total_samples": self.total_samples,
            "total_fields": self.total_fields,
            "allocated_fields": self.allocated_fields,
            "registered_tasks": list(self.consumption_status.keys()),
        }

        if self.production_status is not None:
            produced_samples = torch.any(self.production_status == 1, dim=1).sum().item()
            stats["produced_samples"] = produced_samples
            stats["production_progress"] = produced_samples / self.total_samples if self.total_samples > 0 else 0

            # Field-wise production statistics
            field_stats = {}
            for field_name, field_idx in self.field_name_mapping.items():
                field_produced = (self.production_status[:, field_idx] == 1).sum().item()
                field_stats[field_name] = {
                    "produced_samples": field_produced,
                    "production_progress": field_produced / self.total_samples if self.total_samples > 0 else 0,
                }
            stats["field_statistics"] = field_stats

        # Consumption statistics per task
        consumption_stats = {}
        for task_name, consumption_tensor in self.consumption_status.items():
            consumed_samples = (consumption_tensor == 1).sum().item()
            consumption_stats[task_name] = {
                "consumed_samples": consumed_samples,
                "consumption_progress": consumed_samples / self.total_samples if self.total_samples > 0 else 0,
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
    Dynamic TransferQueue Controller with partition-based data management.

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

    def __init__(self) -> None:
        """Initialize the Dynamic TransferQueue Controller."""
        self.controller_id = f"DYNAMIC_TQ_CONTROLLER_{uuid4()}"

        # Initialize ZMQ sockets for communication
        self._init_zmq_socket()

        # Partition management
        self.partitions: dict[str, DataPartitionStatus] = {}  # partition_id -> DataPartitionStatus

        # Partition GlobalIndex management
        self.index_manager = PartitionIndexManager() # partition_id -> global_indexes

        # Connected storage managers tracking
        self._connected_storage_managers: set[str] = set()

        # Start background processing threads
        self._start_process_handshake()
        self._start_process_update_data_status()
        self._start_process_request()

        logger.info(f"Dynamic TransferQueue Controller {self.controller_id} initialized")

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
    def get_partition_index_range(self, partition) -> set:
        return self.index_manager.get_indexes_for_partition(partition)

    # ==================== Data Production API ====================

    def update_production_status(
        self,
        partition_id: str,
        sample_indices: list[int],
        field_names: list[str],
        dtypes: Optional[dict[int, dict[str, Any]]] = None,
        shapes: Optional[dict[int, dict[str, Any]]] = None,
    ) -> bool:
        """
        Update production status for specific samples and fields in a partition.
        Delegates to the partition's own update_production_status method.

        Args:
            partition_id: ID of the partition
            sample_indices: List of sample indices to update
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

        success = partition.update_production_status(sample_indices, field_names, dtypes, shapes)
        if success:
            logger.debug(
                f"Updated production status for partition {partition_id}: samples={sample_indices}, fields={field_names}"
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

    def generate_data_status_mask(
        self, partition_id: str, field_names: list[str], task_name: str, sample_filter: Optional[list[int]] = None
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate data availability mask for a specific partition.
        Delegates to the partition's own method.

        Args:
            partition_id: ID of the partition
            field_names: List of field names to check
            task_name: Name of the consumer task
            sample_filter: Optional list of specific sample indices to consider

        Returns:
            Tuple of (row_mask, col_mask) tensors if partition exists, (None, None) otherwise
        """
        partition = self.get_partition(partition_id)
        if not partition:
            return None, None

        return partition.generate_data_status_mask(field_names, task_name, sample_filter)

    def get_metadata(
        self,
        data_fields: list[str],
        partition_id: str,
        mode: str = "fetch",
        task_name: str | None = None,
        batch_size: int | None = None,
        get_n_samples=False,  # TODO: get_n_samples作用在哪个步骤？insert模式设置了get_n_samples=True，但是没看到有对应的处理逻辑
        *args,
        **kwargs,
    ) -> BatchMeta:
        """
        Retrieve metadata with support for three modes.

        Args:
            data_fields: List of field names to include in metadata
            batch_size: Number of samples to retrieve
            global_step: Global step for which to retrieve metadata
            mode: Operation mode - 'insert', 'fetch', or 'force_fetch'
                - mode="insert": Insert metadata for new rows (without checking data status)
                - mode="fetch": Retrieve metadata for ready data (check data status and sample)
                - mode="force_fetch": Directly return metadata (without checking data status)
            task_name: Name of the consumer task (required for fetch modes)
            get_n_samples: Whether to retrieve n_samples as groups
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
            # TODO: 区分初次put_data和clear_meta获取batch_global_indices的方法
            if data_fields:
                # 初次put_data时，调用insert模式的get_metadata
                batch_global_indices = self.index_manager.allocate_indexes(partition_id, count=batch_size)
            else:
                # clear metadata时调用get_metadata传入的data_fields为空
                batch_global_indices = self.index_manager.get_indexes_for_partition(partition_id)
            return self.generate_batch_meta(partition_id, batch_global_indices, data_fields, task_name, mode)

        assert task_name is not None
        if mode == "fetch":
            # Find consumable samples within current batch and package into BatchMeta when reading

            start_time = time.time()
            while True:
                ready_for_consume_idx = self.scan_data_status(partition_id, data_fields, task_name, batch_size)

                if len(ready_for_consume_idx) >= batch_size:
                    break

                if time.time() - start_time > TQ_CONTROLLER_GET_METADATA_TIMEOUT:
                    raise TimeoutError(
                        f"Timeout while waiting for sufficient data. "
                        f"Required: {batch_size}, Available: {len(ready_for_consume_idx)}"
                    )

                logger.warning(
                    f"Insufficient data available. Required: {batch_size}, "
                    f"Available: {len(ready_for_consume_idx)}. Retrying in "
                    f"{TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                )
                time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
            logger.debug(f"ready for consume idx: {ready_for_consume_idx}")
            batch_global_indices = ready_for_consume_idx
        elif mode == "force_fetch":
            global_indexes_range = self.index_manager.get_indexes_for_partition(partition_id)
            consumer_status = self.get_consumption_status(partition_id, task_name)
            not_consumed_idx = [i for i in global_indexes_range if consumer_status[i] == 0]
            batch_global_indices = not_consumed_idx

        # # Mark this batch of data as consumed
        # consumer_status = self.get_consumption_status(partition_id, task_name)
        # consumer_status[batch_global_indices] = 1

        # Package into metadata
        metadata = self.generate_batch_meta(partition_id, batch_global_indices, data_fields, task_name, mode)
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
            ready_sample_indices = partition.scan_data_status(data_fields, task_name, sample_filter)

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
        self, partition_id: str, batch_global_indices: list[int], data_fields: list[str], task_name: str, mode: str = "fetch"
    ) -> BatchMeta:
        """
        Generate BatchMeta for specific samples in a partition.

        Args:
            partition_id: ID of the partition
            sample_indices: List of sample indices to include
            field_names: List of field names to include
            task_name: Name of the consumer task
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

        # Mark samples as consumed if in fetch or force_fetch mode
        if mode in ["fetch", "force_fetch"]:
            partition.mark_consumed(task_name, batch_global_indices)

        # Generate sample metadata
        samples = []
        for global_index in batch_global_indices:
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

            # TODO: (baichao) SampleMeta中的global_step替换成partition_id
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
            return False

        global_indexes_range = list(self.index_manager.get_indexes_for_partition(partition_id))
        success = partition.clear_data(global_indexes_range, clear_consumption)
        self.index_manager.release_indexes(partition_id)
        if success:
            logger.info(f"Cleared data for partition {partition_id}")
        return success

    # ==================== ZMQ Communication Methods ====================
    # These methods are largely unchanged from the original implementation

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
                # Handle new partition-based metadata requests
                params = request_msg.body
                partition_id = params.get("partition_id")

                if partition_id:
                    metadata = self.get_metadata(
                        data_fields=params["data_fields"],
                        batch_size=params["batch_size"],
                        partition_id=partition_id,
                        mode=params.get("mode", "fetch"),
                        task_name=params.get("task_name", None),
                        get_n_samples=params.get("get_n_samples", False),
                    )
                else:
                    raise ValueError(f"Please set the correct partition_id, for example: train_$global_step")

                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )

            elif request_msg.request_type == ZMQRequestType.GET_CLEAR_META:
                params = request_msg.body
                # TODO: (baichao) GET_CLEAR_META消息体需要包含partition_id
                partition_id = params.get("partition_id")
                if partition_id:
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
                # TODO: (baichao) CLEAR_META消息体需要包含partition_id
                partition_id = params.get("partition_id")
                if partition_id:
                    self.clear(partition_id)

            elif request_msg.request_type == ZMQRequestType.CHECK_CONSUMPTION:
                # Handle consumption status checks
                params = request_msg.body
                partition_id = params.get("partition_id")

                if partition_id:
                    # New partition-based consumption check
                    consumption_status = self.get_consumption_status(partition_id, params["task_name"])
                    sample_filter = params.get("sample_filter")

                    if consumption_status is not None and sample_filter:
                        batch_status = consumption_status[sample_filter]
                        consumed = torch.all(batch_status == 1).item()
                    elif consumption_status is not None:
                        batch_status = consumption_status
                        consumed = torch.all(batch_status == 1).item()
                    else:
                        consumed = False
                else:
                    raise ValueError(f"Please set the correct partition_id, for example: train_$global_step")

                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={
                        "partition_id": partition_id,
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
                # TODO: (baichao) NOTIFY_DATA_UPDATE消息体需要包含partition_id
                partition_id = message_data.get("partition_id")

                # Update production status
                success = self.update_production_status(
                    partition_id=partition_id,
                    sample_indices=message_data.get("global_indexes", []),
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
