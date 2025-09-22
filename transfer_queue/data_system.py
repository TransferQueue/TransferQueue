import asyncio
import dataclasses
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from functools import wraps
from operator import itemgetter
from threading import Thread
from typing import Any, Callable, Iterator, Optional, Union
from uuid import uuid4

import numpy as np
import ray
import torch
import zmq
import zmq.asyncio
from ray.util import get_node_ip_address
from tensordict import TensorDict
from torch import Tensor

from transfer_queue.utils.utils import (
    ProductionStatus,
    TransferQueueRole,
    random_sampler,
)
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.INFO))

CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT = int(os.environ.get("CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT", 30))
CONTROLLER_DATA_UPDATE_RESPONSE_TIMEOUT = int(os.environ.get("CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT", 600))
POLLER_TIMEOUT_MS = os.environ.get("POLLER_TIMEOUT_MS", 1000)
CONTROLLER_GET_METADATA_TIMEOUT = int(os.environ.get("CONTROLLER_GET_METADATA_TIMEOUT", 300))
CONTROLLER_GET_METADATA_CHECK_INTERVAL = int(os.environ.get("CONTROLLER_GET_METADATA_CHECK_INTERVAL", 1))
INIT_FIELD_NUM = os.environ.get("INIT_FIELD_NUM", 10)


@dataclass
class FieldMeta:
    """
    Records the metadata of a single data field. (name, dtype, shape, etc.)
    """

    # field name (e.g., 'prompt', 'response', etc.)
    name: str

    # data schema info
    dtype: Optional[torch.dtype]  # e.g., torch.float32, torch.int64, etc.
    shape: Optional[torch.Size]  # e.g., torch.Size([seq_len]), torch.Size([seq_len, feature_dim]), etc.

    # data status info
    production_status: ProductionStatus = ProductionStatus.NOT_PRODUCED  # production status for this field

    def __str__(self) -> str:
        return (
            f"FieldMeta(name='{self.name}', dtype={self.dtype}, "
            f"shape={self.shape}, production_status={self.production_status})"
        )

    @property
    def is_ready(self) -> bool:
        """Check if this field is ready for consumption"""
        return self.production_status == ProductionStatus.READY_FOR_CONSUME

    def equals(self, other: "FieldMeta") -> bool:
        """Check if two FieldMeta are equal (based on name, dtype, shape)"""
        return self.name == other.name and self.dtype == other.dtype and self.shape == other.shape


@dataclass
class SampleMeta:
    """
    Records the metadata of a single data sample (stored as a row in the data system).
    """

    # algorithm related info
    global_step: int  # global step, used for data versioning

    # data retrival info
    global_index: int  # global row index, uniquely identifies a data sample
    storage_id: str  # storage unit id
    local_index: int  # local row index in the storage unit

    # data fields info
    # this fields may not contain all the fields of the sample, but only fields-of-interest
    fields: dict[str, FieldMeta]

    def __post_init__(self):
        """Initialize is_ready property based on field readiness"""
        # Check if all fields are ready and update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))

    def __str__(self) -> str:
        return (
            f"SampleMeta(global_step={self.global_step}, "
            f"global_index={self.global_index}, storage_id='{self.storage_id}', "
            f"local_index={self.local_index}, fields={self.fields})"
        )

    @property
    def field_names(self) -> list[str]:
        """Get list of field names for this sample"""
        return list(self.fields.keys())

    @property
    def batch_index(self) -> int:
        """Get the batch index of this sample (to be set by BatchMeta)"""
        return getattr(self, "_batch_index", -1)

    def get_field_by_name(self, name: str) -> Optional[FieldMeta]:
        """Get FieldMeta by field name"""
        return self.fields.get(name)

    def has_field(self, name: str) -> bool:
        """Check if this sample has a specific field"""
        return name in self.fields

    def is_field_ready(self, field_name: str) -> bool:
        """Check if a specific field is ready for consumption"""
        field = self.fields.get(field_name)
        return field.is_ready if field else False

    def add_fields(self, names: list[str], dtypes: list[torch.dtype], shapes: list[torch.Size]) -> None:
        """
        Add new fields to this sample. New fields will be initialized with default FieldMeta.
        This modifies the sample in-place to include the new fields.
        """
        for field_name, dtype, shape in zip(names, dtypes, shapes, strict=False):
            assert field_name not in self.fields, f"Field '{field_name}' already exists in SampleMeta."
            self.fields[field_name] = FieldMeta(name=field_name, dtype=dtype, shape=shape)
        # Update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))

    def union(self, other: "SampleMeta", validate: bool = True) -> "SampleMeta":
        """
        Create a union of this sample's fields with another sample's fields.
        Assume both samples have the same global index, and if fields overlap, they must be identical.
        Args:
            other: Another SampleMeta to union with
            validate: Whether to validate union conditions
        Returns:
            New SampleMeta with unioned fields (None if validation fails)
        """
        if validate:
            if self.global_index != other.global_index:
                raise ValueError(
                    f"Error: Global indexes ({self.global_index} and {other.global_index}) do not match for union."
                )

        # Merge fields
        merged_fields = _union_fields(self.fields, other.fields)
        self.fields = merged_fields

        # Update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))
        return self

    @property
    def is_ready(self) -> bool:
        """Check if all fields in this sample are ready for consumption"""
        return getattr(self, "_is_ready", False)

    @property
    def production_status(self) -> dict[str, ProductionStatus]:
        """Get production status for all fields (backward compatibility)"""
        return {name: field.production_status for name, field in self.fields.items()}


@dataclass
class StorageMetaGroup:
    """
    Represents a group of samples stored in the same storage unit.

    This is an optimized implementation that only stores SampleMeta objects,
    eliminating the need for separate global_indexes, local_indexes, and field_names
    fields since all this information is already available in the SampleMeta objects.

    This approach reduces memory usage and improves performance by avoiding data
    duplication while providing all necessary functionality for AsyncTransferQueueClient
    operations.
    """

    storage_id: str
    sample_metas: list[SampleMeta] = dataclasses.field(default_factory=list)

    def add_sample_meta(self, sample_meta: SampleMeta) -> None:
        """Add a SampleMeta object to this storage group"""
        self.sample_metas.append(sample_meta)

    def get_batch_indexes(self) -> list[int]:
        """Get all internal indexes from stored SampleMeta objects"""
        return [meta.batch_index for meta in self.sample_metas]

    def get_global_indexes(self) -> list[int]:
        """Get all global indexes from stored SampleMeta objects"""
        return [meta.global_index for meta in self.sample_metas]

    def get_local_indexes(self) -> list[int]:
        """Get all local indexes from stored SampleMeta objects"""
        return [meta.local_index for meta in self.sample_metas]

    def get_field_names(self) -> list[str]:
        """Get all unique field names from stored SampleMeta objects"""
        all_fields: set[str] = set()
        for meta in self.sample_metas:
            all_fields.update(meta.fields.keys())
        return list(all_fields)

    def get_transfer_info(self, field_names: Optional[list[str]] = None) -> dict[str, list | dict]:
        """Convert to dictionary format for backward compatibility"""
        if field_names is None:
            field_names = self.get_field_names()
        return {
            "batch_indexes": self.get_batch_indexes(),
            "global_indexes": self.get_global_indexes(),
            "local_indexes": self.get_local_indexes(),
            "fields": field_names,
            "field_data": {},  # Placeholder for field data to be filled later
        }

    @property
    def size(self) -> int:
        """Number of samples in this storage meta group"""
        return len(self.sample_metas)

    @property
    def is_empty(self) -> bool:
        """Check if this storage meta group is empty"""
        return len(self.sample_metas) == 0

    def __len__(self) -> int:
        """Number of samples in this storage meta group"""
        return self.size

    def __bool__(self) -> bool:
        """Truthiness based on whether group has samples"""
        return not self.is_empty

    def __str__(self) -> str:
        return f"StorageMetaGroup(storage_id='{self.storage_id}', size={self.size})"


@dataclass
class BatchMeta:
    """
    Records the metadata of a batch of data samples.

    Supports iteration and indexing:
    - Iteration: for sample in batch:
    - Length: count = len(batch)
    - Indexing: first_sample = batch[0]
    """

    samples: list[SampleMeta]
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Initialize all computed properties during initialization"""
        # Basic properties
        object.__setattr__(self, "_size", len(self.samples))
        object.__setattr__(self, "_is_ready", all(sample.is_ready for sample in self.samples))

        # Pre-compute all list properties for better performance
        if self.samples:
            for idx, sample in enumerate(self.samples):
                object.__setattr__(sample, "_batch_index", idx)  # Ensure batch_index is set correctly

            object.__setattr__(self, "_global_indexes", [sample.global_index for sample in self.samples])
            object.__setattr__(self, "_local_indexes", [sample.local_index for sample in self.samples])
            object.__setattr__(self, "_storage_ids", [sample.storage_id for sample in self.samples])

            # assume all samples have the same fields.
            object.__setattr__(self, "_fields", self.samples[0].field_names)

            # Initialize storage groups for efficient client operations
            storage_meta_groups = self._build_storage_meta_groups()
            object.__setattr__(self, "_storage_meta_groups", storage_meta_groups)
        else:
            object.__setattr__(self, "_global_indexes", [])
            object.__setattr__(self, "_local_indexes", [])
            object.__setattr__(self, "_storage_ids", [])
            object.__setattr__(self, "_fields", [])
            object.__setattr__(self, "_storage_meta_groups", {})

    @property
    def size(self) -> int:
        """Return the number of samples in this batch"""
        return getattr(self, "_size", 0)

    @property
    def global_indexes(self) -> list[int]:
        """Get all global indexes in this batch"""
        return getattr(self, "_global_indexes", [])

    @property
    def fields(self) -> list[str]:
        """Get all unique field names in this batch"""
        return getattr(self, "_fields", [])

    @property
    def local_indexes(self) -> list[int]:
        """Get all local indexes in this batch"""
        return getattr(self, "_local_indexes", [])

    @property
    def storage_ids(self) -> list[str]:
        """Get all storage unit IDs in this batch"""
        return getattr(self, "_storage_ids", [])

    @property
    def is_ready(self) -> bool:
        """Check if all samples in this batch are ready for consumption"""
        # TODO: get ready status from controller realtime
        return getattr(self, "_is_ready", False)

    def _build_storage_meta_groups(self) -> dict[str, StorageMetaGroup]:
        """Build storage groups from samples during initialization"""
        storage_meta_groups: dict[str, StorageMetaGroup] = {}

        for sample in self.samples:
            storage_id = sample.storage_id
            if storage_id not in storage_meta_groups:
                storage_meta_groups[storage_id] = StorageMetaGroup(storage_id=storage_id)

            # Use add_sample_meta to store SampleMeta references directly
            storage_meta_groups[storage_id].add_sample_meta(sample)

        return storage_meta_groups

    @property
    def storage_meta_groups(self) -> dict[str, StorageMetaGroup]:
        """Get storage groups organized by storage_id"""
        return getattr(self, "_storage_meta_groups", {})

    @property
    def storage_unit_ids(self) -> list[str]:
        """Get list of all storage unit IDs"""
        return list(self.storage_meta_groups.keys())

    def get_storage_meta_groups(self, storage_id: str) -> Optional[StorageMetaGroup]:
        """Get storage group by storage ID"""
        return self.storage_meta_groups.get(storage_id)

    # Extra info interface methods
    def get_extra_info(self, key: str, default: Any = None) -> Any:
        """Get extra info by key"""
        return self.extra_info.get(key, default)

    def set_extra_info(self, key: str, value: Any) -> None:
        """Set extra info by key"""
        self.extra_info[key] = value

    def update_extra_info(self, info_dict: dict[str, Any]) -> None:
        """Update extra info with multiple key-value pairs"""
        self.extra_info.update(info_dict)

    def remove_extra_info(self, key: str) -> Any:
        """Remove extra info by key and return its value"""
        return self.extra_info.pop(key, None)

    def clear_extra_info(self) -> None:
        """Clear all extra info"""
        self.extra_info.clear()

    def has_extra_info(self, key: str) -> bool:
        """Check if extra info contains a specific key"""
        return key in self.extra_info

    def get_all_extra_info(self) -> dict[str, Any]:
        """Get all extra info as a dictionary"""
        return self.extra_info.copy()

    def add_fields(self, names: list[str], **kwargs) -> None:
        """
        Add new fields to all samples in this batch. New fields will be initialized with default FieldMeta.
        This modifies each sample in-place to include the new fields.
        """
        for sample in self.samples:
            sample.add_fields(names, **kwargs)

        # Update batch-level fields cache
        updated_fields = set(self.fields)
        updated_fields.update(names)
        object.__setattr__(self, "_fields", list(updated_fields))

    def __iter__(self) -> Iterator[SampleMeta]:
        """Iterate over samples in this batch."""
        # TODO：考虑返回SampleMeta是否合适？还是返回一个只有一个SampleMeta的BatchMeta
        return iter(self.samples)

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return len(self.samples)

    def __getitem__(self, item):
        if isinstance(item, int | np.integer):
            sample_meta = self.samples[item] if self.samples else []
            return BatchMeta(samples=[sample_meta], extra_info=self.extra_info)
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported now!")

    def chunk(self, chunks: int) -> list["BatchMeta"]:
        """
        Split this batch into smaller chunks.

        Args:
            chunks: number of chunks

        Return:
            List of smaller BatchMeta chunks

        """
        chunk_list = []
        n = len(self.samples)

        # Calculate the base size and remainder of each chunk
        base_size = n // chunks
        remainder = n % chunks

        start = 0
        for i in range(chunks):
            # Calculate the size of the current chunk(the first remainder chunk is 1 more than the base size)
            current_chunk_size = base_size + 1 if i < remainder else base_size
            end = start + current_chunk_size
            chunk_samples = self.samples[start:end]
            chunk = BatchMeta(samples=chunk_samples, extra_info=self.extra_info)
            chunk_list.append(chunk)
            start = end
        return chunk_list

    @classmethod
    def concat(cls, data: list["BatchMeta"], validate: bool = True) -> Optional["BatchMeta"]:
        """
        Concatenate multiple BatchMeta chunks into one large batch.

        Args:
            data: List of BatchMeta chunks to concatenate
            validate: Whether to validate concatenation conditions

        Returns:
            Concatenated BatchMeta (None if validation fails)
        """
        if not data:
            return None

        if validate:
            base_fields = data[0].fields

            for chunk in data:
                if chunk.fields != base_fields:
                    raise ValueError("Error: Field names do not match for concatenation.")

        # Combine all samples
        all_samples = []
        for chunk in data:
            all_samples.extend(chunk.samples)

        # Combine extra info (later chunks overwrite earlier ones on key conflicts)
        all_extra_info = {}
        for chunk in data:
            all_extra_info.update(chunk.extra_info)
        # TODO: can return data[0].extra_info directly
        return BatchMeta(samples=all_samples, extra_info=all_extra_info)

    def union(self, other: "BatchMeta", validate: bool = True) -> Optional["BatchMeta"]:
        """
        Create a union of this batch's fields with another batch's fields.
        Assume both batches have the same global indices.
        Args:
            other: Another BatchMeta to union with
            validate: Whether to validate union conditions
        Returns:
            New BatchMeta with unioned fields (None if validation fails)
        """
        if validate:
            if self.size != other.size:
                raise ValueError("Error: Batch sizes do not match for union.")

            self_global_indexes = sorted(self.global_indexes)
            other_global_indexes = sorted(other.global_indexes)
            if self_global_indexes != other_global_indexes:
                raise ValueError("Error: Global indexes do not match for union.")

        # Create a mapping from global_index to SampleMeta in the other batch
        other_sample_map = {sample.global_index: sample for sample in other.samples}

        # Merge samples
        merged_samples = []
        for sample in self.samples:
            if sample.global_index in other_sample_map:
                other_sample = other_sample_map[sample.global_index]
                merged_sample = sample.union(other_sample, validate=validate)
                merged_samples.append(merged_sample)
            else:
                merged_samples.append(sample)

        # Merge extra info dictionaries
        merged_extra_info = {**self.extra_info, **other.extra_info}

        return BatchMeta(samples=merged_samples, extra_info=merged_extra_info)

    @classmethod
    def from_samples(
        cls, samples: SampleMeta | list[SampleMeta], extra_info: Optional[dict[str, Any]] = None
    ) -> "BatchMeta":
        """
        Create a BatchMeta from a single SampleMeta or a list of SampleMeta objects.

        Args:
            samples: A single SampleMeta or a list of SampleMeta objects
            extra_info: Optional additional information to store with the batch

        Returns:
            BatchMeta instance containing the provided sample(s)

        Example:
            >>> sample_meta = SampleMeta(...)
            >>> batch_meta = BatchMeta.from_samples(sample_meta)

            >>> sample_metas = [sample1, sample2, sample3]
            >>> batch_meta = BatchMeta.from_samples(sample_metas, extra_info={"source": "training"})
        """
        if extra_info is None:
            extra_info = {}

        if isinstance(samples, SampleMeta):
            samples = [samples]

        return cls(samples=samples, extra_info=extra_info)

    @classmethod
    def empty(cls, extra_info: Optional[dict[str, Any]] = None) -> "BatchMeta":
        """
        Create an empty BatchMeta with no samples.

        Args:
            extra_info: Optional additional information to store with the batch

        Returns:
            Empty BatchMeta instance

        Example:
            >>> empty_batch = BatchMeta.empty()
        """
        if extra_info is None:
            extra_info = {}
        return cls(samples=[], extra_info=extra_info)


@ray.remote(num_cpus=1)
class TransferQueueController:
    def __init__(
        self,
        num_storage_units: int,
        global_batch_size: int,
        num_global_batch: int = 1,
        num_n_samples: int = 1,
    ) -> None:
        self.controller_id = f"TQ_CONTROLLER_{uuid4()}"

        self._init_zmq_socket()  # 通过ZMQ实现数据通信

        self.num_storage_units = num_storage_units
        self.global_batch_size = global_batch_size  # 用作global index的offset，区分是哪个global step对应的数据
        self.num_global_batch = num_global_batch
        self.num_n_samples = num_n_samples
        self.total_storage_size = self.global_batch_size * self.num_global_batch * self.num_n_samples

        self.data_production_status = torch.zeros(
            self.total_storage_size, INIT_FIELD_NUM, dtype=torch.int8
        )  # 默认初始化20个字段，可动态扩展
        # task_name -> consumption_status
        self.data_consumption_status: dict[str, torch.Tensor] = {}
        self.field_name_mapping: dict[str, int] = {}  # 一个data_field和data_status列index的映射表
        # Per-sample dtype and shape storage: {global_index: {field_name: {'dtype': dtype, 'shape': shape}}}
        self.per_tensor_dtype_mapping: dict[int, dict[str, torch.dtype]] = {}
        self.per_tensor_shape_mapping: dict[int, dict[str, torch.Size]] = {}

        self._build_index_storage_mapping()

        self._start_process_handshake()
        self._start_process_update_data_status()
        self._start_process_request()

    def _get_consumer_status(self, task_name: str) -> torch.Tensor:
        # 获取或创建指定消费者的消费状态张量
        if task_name not in self.data_consumption_status:
            # 为新消费者初始化状态
            self.data_consumption_status[task_name] = torch.zeros(self.total_storage_size, dtype=torch.int8)
        return self.data_consumption_status[task_name]

    def _get_per_tensor_dtype(self, global_index: int, field_name: str) -> Optional[torch.dtype]:
        """Get dtype for a specific sample and field."""
        return self.per_tensor_dtype_mapping.get(global_index, {}).get(field_name)

    def _get_per_tensor_shape(self, global_index: int, field_name: str) -> Optional[torch.Size]:
        """Get shape for a specific sample and field."""
        return self.per_tensor_shape_mapping.get(global_index, {}).get(field_name)

    def _step_to_global_index_range(self, global_step: int) -> tuple[int, int]:
        start_idx = (global_step % self.num_global_batch) * self.global_batch_size * self.num_n_samples
        end_idx = start_idx + self.global_batch_size * self.num_n_samples

        return start_idx, end_idx

    def generate_data_status_mask(
        self, data_fields: list[str], global_step: int, task_name: str
    ) -> tuple[Tensor, Tensor]:
        # 该函数在_get_meta中被调用，根据用户指定的字段和当前的step，生成一个mask矩阵
        # 其中用户指定的字段为入参，当前step对应的行（即global index范围）按照顺序映射即可
        # 该mask矩阵将self.data_production_status中，用户需要的行列选中，
        # 同时将self.data_consumption_status反选，从而生成一个子矩阵，
        # 以便在_get_meta的过程中支持自动向量化操作加速状态查询（直接按行sum判断是否等于shape[1]即可）

        # 检查所有请求的字段是否已注册
        for col in data_fields:
            if col not in self.field_name_mapping:
                # 如果有未注册的列，返回空掩码表示没有可用数据
                empty_row_mask = torch.zeros(self.data_production_status.shape[0], dtype=torch.bool)
                empty_col_mask = torch.zeros(self.data_production_status.shape[1], dtype=torch.bool)
                return empty_row_mask, empty_col_mask

        # step映射到global index
        start_idx, end_idx = self._step_to_global_index_range(global_step)
        row_mask = torch.zeros(self.data_production_status.shape[0], dtype=torch.bool)
        row_mask[start_idx:end_idx] = True

        # 按消费状态反选
        consumer_status = self._get_consumer_status(task_name)
        unconsumed_mask = consumer_status == 0
        row_mask &= unconsumed_mask

        # 选中指定的字段
        col_mask = torch.zeros(self.data_production_status.shape[1], dtype=torch.bool)
        valid_fields = [self.field_name_mapping[col] for col in data_fields]
        if valid_fields:
            col_mask[valid_fields] = True

        return row_mask, col_mask

    def _build_index_storage_mapping(self):
        # 根据数据系统总空间与StorageUnit数量，划分每个Sample应该存储的位置，
        # 并维护global index和每个存储内local index的映射

        # 为每条样本分配存储节点；注意我们应该将每个GBS数据打散在不同存储节点上。
        # 这里和generate_data_status_mask一样，默认按照顺序排列样本
        real_global_batch_size = self.global_batch_size * self.num_n_samples
        global_batch_per_storage_unit = math.ceil(real_global_batch_size / self.num_storage_units)

        # 构建global index与storage unit之间的映射，用于查找每条数据对应的存储节点位置
        batch_storage_indices = np.repeat(np.arange(self.num_storage_units), global_batch_per_storage_unit)[
            :real_global_batch_size
        ]
        self._global_index_storage_rank_mapping = np.tile(batch_storage_indices, self.num_global_batch)

        # 构建global index与每个storage unit之间local index之间的映射
        indices = np.arange(self.total_storage_size)
        pos_in_batch = indices % real_global_batch_size
        g = indices // real_global_batch_size
        pos_in_block = pos_in_batch % global_batch_per_storage_unit
        self.global_index_local_index_mapping = g * global_batch_per_storage_unit + pos_in_block

    def get_data_production_status(self) -> torch.Tensor:
        return self.data_production_status

    def get_field_name_mapping(self) -> dict[str, Any]:
        return self.field_name_mapping

    def get_data_consumption_status(self) -> dict[str, torch.Tensor]:
        return self.data_consumption_status

    def get_global_index_mapping(self):
        return self._global_index_storage_rank_mapping, self.global_index_local_index_mapping

    def _get_metadata(
        self,
        data_fields: list[str],
        batch_size: int,
        mode: str = "fetch",
        global_step=0,
        task_name: str | None = None,
        get_n_samples=False,
        *args,
        **kwargs,
    ) -> BatchMeta:
        """
        获取元数据，支持两种模式：
        - mode="insert": 插入新行的元数据（不检查数据状态）
        - mode="fetch": 获取已就绪的元数据（检查数据状态并采样）
        - mode="force_fetch": 直接返回元数据（不检查数据状态）
        """
        if mode == "insert":
            # TODO 当前仅支持put整个gbs的数据，后续待扩展支持多次put到同一step
            assert batch_size == self.global_batch_size
            start_idx, end_idx = self._step_to_global_index_range(global_step)
            batch_global_indexes = list(range(start_idx, end_idx))
            return self._generate_batch_meta(global_step, batch_global_indexes, data_fields, mode)

        assert task_name is not None
        if mode == "fetch":
            # 向TransferQueue读数据时，查找当前batch内可被消费的样本，并打包返回BatchMeta

            # 循环检查可被消费的数据
            start_time = time.time()
            while True:
                ready_for_consume_idx = self._scan_data_status(data_fields, global_step, task_name, get_n_samples)

                if len(ready_for_consume_idx) >= batch_size:
                    break

                if time.time() - start_time > CONTROLLER_GET_METADATA_TIMEOUT:
                    raise TimeoutError(
                        f"Timeout while waiting for sufficient data. "
                        f"Required: {batch_size}, Available: {len(ready_for_consume_idx)}"
                    )

                logger.warning(
                    f"Insufficient data available. Required: {batch_size}, "
                    f"Available: {len(ready_for_consume_idx)}. Retrying in {CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                )
                time.sleep(CONTROLLER_GET_METADATA_CHECK_INTERVAL)
            logger.debug(f"ready for consume idx: {ready_for_consume_idx}")

            batch_global_indexes = random_sampler(ready_for_consume_idx, batch_size, get_n_samples, self.num_n_samples)
        elif mode == "force_fetch":
            start_idx, end_idx = self._step_to_global_index_range(global_step)
            consumer_status = self._get_consumer_status(task_name)
            not_consumed_idx = [i for i in range(start_idx, end_idx) if consumer_status[i] == 0]
            batch_global_indexes = random_sampler(not_consumed_idx, batch_size, get_n_samples, self.num_n_samples)

        # 标记这批数据状态为已消费
        consumer_status = self._get_consumer_status(task_name)
        consumer_status[batch_global_indexes] = 1
        # 打包为metadata
        metadata = self._generate_batch_meta(global_step, batch_global_indexes, data_fields, mode)
        # 6. 如果是方式2，则将metadata进行缓存
        # if dp_rank and dp_size and rank_id:
        #     pass
        logger.debug(f"_get_metadata: {metadata}")

        return metadata

    def _scan_data_status(
        self, data_fields: list[str], global_step: int, task_name: str, get_n_samples: bool
    ) -> list[int]:
        # 获取行和列掩码
        row_mask, col_mask = self.generate_data_status_mask(data_fields, global_step, task_name)
        logger.debug(f"row_mask, col_mask: {row_mask, col_mask}")

        if not row_mask.any() or not col_mask.any():
            return []

        # 提取关注的数据状态子集
        logger.debug(f"self.data_production_status: {self.data_production_status}")
        data_status_of_interest = self.data_production_status[:, col_mask]
        logger.debug(f"data_status_of_interest: {data_status_of_interest}")

        # 使用torch.all向量化检查替代求和比较
        all_fields_ready = torch.all(data_status_of_interest, dim=1)

        # 结合行掩码筛选符合条件的样本
        ready_mask = all_fields_ready & row_mask

        if get_n_samples and self.num_n_samples > 1:
            # 重塑为组视图并检查组完整性
            group_all_ready = torch.all(ready_mask.view(-1, self.num_n_samples), dim=1)

            # 获取完整就绪的组索引
            ready_group_indices = group_all_ready.nonzero(as_tuple=False).flatten()

            # 计算所有样本索引
            sample_offset = torch.arange(self.num_n_samples)
            ready_for_consume_idx = (
                (ready_group_indices.unsqueeze(1) * self.num_n_samples + sample_offset).flatten().tolist()
            )

            return ready_for_consume_idx
        else:
            ready_for_consume_idx = torch.nonzero(ready_mask, as_tuple=False).flatten().tolist()
            logger.debug(f"ready_for_consume_idx: {ready_for_consume_idx}")

            return ready_for_consume_idx

    def _generate_batch_meta(
        self, global_step: int, global_indexes: list[int], data_fields: list[str], mode: str
    ) -> BatchMeta:
        # 根据给定的global index，查找self.global_index_local_index_mapping
        # 和self._global_index_storage_id_mapping，确定对应存储节点的地址，并构建BatchMeta
        global_arr = np.array(global_indexes)
        storage_ids = self.global_index_storage_id_mapping[global_arr]
        local_indexes = self.global_index_local_index_mapping[global_arr]

        samples = []

        # Create samples from the flattened BatchMeta data
        # TODO: 待优化
        for i, global_index in enumerate(global_indexes):
            local_index = local_indexes[i]
            storage_id = storage_ids[i]

            # Create FieldMeta objects for each field
            fields = []
            for field_name in data_fields:
                if mode == "fetch":
                    production_status = ProductionStatus.READY_FOR_CONSUME  # Since we filtered by ready status
                    # Get per-tensor dtype and shape for this specific global_index and field
                    dtype = self._get_per_tensor_dtype(global_index, field_name)
                    shape = self._get_per_tensor_shape(global_index, field_name)
                elif mode == "insert":
                    production_status = ProductionStatus.NOT_PRODUCED  # FIXME: not real-time
                    dtype = None
                    shape = None
                elif mode == "force_fetch":
                    col_index = self.field_name_mapping.get(field_name)
                    if col_index is not None and self.data_production_status[global_index, col_index] == 1:
                        production_status = ProductionStatus.READY_FOR_CONSUME
                        dtype = self._get_per_tensor_dtype(global_index, field_name)
                        shape = self._get_per_tensor_shape(global_index, field_name)
                    else:
                        production_status = ProductionStatus.NOT_PRODUCED
                        dtype = None
                        shape = None
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=dtype,
                    shape=shape,
                    production_status=production_status,
                )
                fields.append(field_meta)

            sample = SampleMeta(
                global_step=global_step,
                global_index=global_index,
                storage_id=storage_id,
                local_index=local_index,
                fields={field.name: field for field in fields},
            )
            samples.append(sample)

        return BatchMeta(samples=samples)

    def _update_production_status(self, indexes: list[int], fields: list[str]) -> None:
        # TODO replace the self.data_production_status == 0 or ==1 operation by using ProductionStatus
        # 更新数据生产状态矩阵
        new_fields = [field for field in fields if field not in self.field_name_mapping]
        if new_fields:
            needed_fields = len(new_fields)
            current_fields = self.data_production_status.shape[1]
            # 扩容数据状态矩阵
            if len(self.field_name_mapping) + needed_fields > current_fields:
                add_fields = max(INIT_FIELD_NUM, needed_fields + 1)
                new_matrix = torch.zeros((self.total_storage_size, add_fields), dtype=torch.int8)
                self.data_production_status = torch.cat([self.data_production_status, new_matrix], dim=1)

        for field in fields:
            if field not in self.field_name_mapping.keys():
                self.field_name_mapping[field] = len(self.field_name_mapping)
        self.data_production_status[
            torch.tensor(indexes)[:, None], torch.tensor([self.field_name_mapping.get(field) for field in fields])
        ] = 1

    def _update_field_info(
        self,
        fields: list[str],
        per_tensor_dtypes: dict[int, dict[str, Any]],
        per_tensor_shapes: dict[int, dict[str, Any]],
        global_indexes: list[int],
    ) -> None:
        """
        Store per-tensor dtype and shape information.

        Args:
            fields: List of field names
            per_tensor_dtypes: Dict mapping global_index to field dtypes {global_index: {field: dtype}}
            per_tensor_shapes: Dict mapping global_index to field shapes {global_index: {field: shape}}
            global_indexes: List of global indexes corresponding to the samples
        """
        for global_idx in global_indexes:
            if global_idx not in self.per_tensor_dtype_mapping:
                self.per_tensor_dtype_mapping[global_idx] = {}
            if global_idx not in self.per_tensor_shape_mapping:
                self.per_tensor_shape_mapping[global_idx] = {}

            for field in fields:
                if global_idx in per_tensor_dtypes and field in per_tensor_dtypes[global_idx]:
                    self.per_tensor_dtype_mapping[global_idx][field] = per_tensor_dtypes[global_idx][field]
                if global_idx in per_tensor_shapes and field in per_tensor_shapes[global_idx]:
                    self.per_tensor_shape_mapping[global_idx][field] = per_tensor_shapes[global_idx][field]

    def _init_zmq_socket(self):
        # 建立3个ZMQ服务端口，分别用于 ①注册发现 ② 接收Client的数据读写请求 ③ 接收Storage发送的状态更新信号
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

        self.zmq_server_info = ZMQServerInfo.create(
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
        # 等待所有存储实例握手;client无需握手以支持动态扩缩容
        # 参考zmq_communication.py中的实现
        # TODO(zjj): 考虑是否需要重传（假设存在Storage没有收到ACK的情况）
        connected_storage_units = set()
        while len(connected_storage_units) < self.num_storage_units:
            identity, serialized_msg = self.handshake_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)
            if request_msg.request_type == ZMQRequestType.HANDSHAKE:
                connected_storage_units.add(request_msg.sender_id)
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.HANDSHAKE_ACK,
                    sender_id=self.controller_id,
                    body={},
                ).serialize()
                self.handshake_socket.send_multipart([identity, response_msg])
                logger.info("Controller send handshake ack successful!")
        self.global_index_storage_id_mapping = np.array(list(connected_storage_units))[
            self._global_index_storage_rank_mapping
        ]
        self.handshake_done.set()

    def _start_process_handshake(self):
        self.handshake_done = threading.Event()
        self.wait_connection_thread = Thread(
            target=self._wait_connection, name="TransferQueueControllerWaitConnectionThread", daemon=True
        )
        self.wait_connection_thread.start()

    def _start_process_update_data_status(self):
        self.process_update_data_status_thread = Thread(
            target=self._update_data_status, name="TransferQueueControllerProcessUpdateDataStatusThread", daemon=True
        )
        self.process_update_data_status_thread.start()

    def _start_process_request(self):
        self.process_request_thread = Thread(
            target=self._process_request, name="TransferQueueControllerProcessRequestThread", daemon=True
        )
        self.process_request_thread.start()

    def _process_request(self):
        # 包含_get_meta、查询当前iteration是否消费完毕等
        self.handshake_done.wait()
        while True:
            # ROUTER套接字接收多部分消息
            identity, serialized_msg = self.request_handle_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)

            if request_msg.request_type == ZMQRequestType.GET_META:
                params = request_msg.body
                logger.info("Controller prepare get metadata...")
                metadata = self._get_metadata(
                    data_fields=params["data_fields"],
                    batch_size=params["batch_size"],
                    global_step=params["global_step"],
                    mode=params.get("mode", "fetch"),
                    task_name=params.get("task_name", None),
                    get_n_samples=params.get("get_n_samples", False),
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )
            elif request_msg.request_type == ZMQRequestType.GET_CLEAR_META:
                params = request_msg.body
                metadata = self._get_metadata(
                    data_fields=[],
                    batch_size=self.global_batch_size,
                    global_step=params["global_step"],
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
                self.clear(global_step=params["global_step"])
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CLEAR_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"message": f"Clear operation completed by controller {self.controller_id}"},
                )
            elif request_msg.request_type == ZMQRequestType.CHECK_CONSUMPTION:
                # 消费状态检查
                params = request_msg.body
                global_step = params["global_step"]

                consumer_status = self._get_consumer_status(params["task_name"])
                start_idx, end_idx = self._step_to_global_index_range(global_step)
                batch_status = consumer_status[start_idx:end_idx]
                consumed = torch.all(batch_status == 1).item()

                # 构建响应消息
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={
                        "global_step": global_step,
                        "consumed": consumed,
                    },
                )
            self.request_handle_socket.send_multipart([identity, response_msg.serialize()])
            logger.debug("Controller request_handle_socket send_multipart successful!")

    def _update_data_status(self):
        # 用于接受来自storage的数据状态更新信息
        while True:
            logger.debug("Prepare _update_data_status...")
            identity, serialized_msg = self.data_status_update_socket.recv_multipart()
            logger.debug("Controller recv update_data_status request!")
            request_msg = ZMQMessage.deserialize(serialized_msg)
            logger.debug(f"[{self.controller_id}]: Controller recv update_data_status request_msg: {request_msg}")

            if request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE:
                message_data = request_msg.body

                fields = message_data.get("fields", [])
                global_indexes = message_data.get("global_indexes", [])
                per_tensor_dtypes = message_data.get("dtypes", {})  # Now a dict of lists
                per_tensor_shapes = message_data.get("shapes", {})  # Now a dict of lists
                # 更新数据生产状态
                logger.debug(f"global_indexes, fields: {global_indexes, fields}")
                self._update_production_status(global_indexes, fields)
                self._update_field_info(fields, per_tensor_dtypes, per_tensor_shapes, global_indexes)
                logger.info("Controller update production status successful!")

                # 发送确认响应
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Data update acknowledged from controller {self.controller_id}",
                    },
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])
                logger.info("Controller send DATA_UPDATE_ACK successful!")
            elif request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR:
                # 处理数据更新错误
                error_msg = request_msg.body.get("message", "Unknown error")
                print(f"Data update error from storage: {error_msg}")

                # 发送错误确认响应
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Error notification acknowledged from controller {self.controller_id}",
                    },
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])

    def get_zmq_server_info(self) -> ZMQServerInfo:
        return self.zmq_server_info

    def clear(self, global_step: int):
        # 清空对应global batch的数据，当前仅实现单个gbs

        start_idx, end_idx = self._step_to_global_index_range(global_step)

        self.data_production_status[start_idx:end_idx, :] = 0
        for task_name in self.data_consumption_status:
            self.data_consumption_status[task_name][start_idx:end_idx] = 0


class StorageUnitData:
    """
    Class used for storing several elements, each element is composed of several fields and corresponding data, like:
    #####################################################
    # local_index | field_name1 | field_name2 | ...   #
    # 0           | item1       | item2       | ...   #
    # 1           | item3       | item4       | ...   #
    # 2           | item5       | item6       | ...   #
    #####################################################
    """

    def __init__(self, storage_size: int):
        # Dict containing field names and corresponding data in the field, e.g. {"field_name1": [data1, data2, ...]}
        self.field_data: dict[str, list] = {}

        # Maximum number of elements stored in storage unit
        self.storage_size = storage_size

    def get_data(self, fields: list[str], local_indexes: list[int]) -> TensorDict[str, list]:
        """
        Get data from storage unit according to given fields and local_indexes.

        param:
            fields: Field names used for getting data.
            local_indexes: Local indexes used for getting data.
        return:
            TensorDict with field names as keys, corresponding data list as values.
        """
        result: dict[str, list] = {}

        for field in fields:
            # Validate field name
            if field not in self.field_data:
                raise ValueError(
                    f"StorageUnitData get_data operation receive invalid field: {field} beyond {self.field_data.keys()}"
                )

            if len(local_indexes) == 1:
                # The unsqueeze op make the shape from n to (1, n)
                result[field] = torch.tensor(list(self.field_data[field][local_indexes[0]])).unsqueeze(0)
            else:
                result[field] = torch.stack(list(itemgetter(*local_indexes)(self.field_data[field])))

        return TensorDict(result)

    def put_data(self, field_data: TensorDict[str, list], local_indexes: list[int]) -> None:
        """
        Put or update data into storage unit according to given field_data and local_indexes.

        param:
            field_data: Dict with field names as keys, corresponding data in the field as values.
            local_indexes: Local indexes used for putting data.
        """
        for f in field_data.keys():
            for i, idx in enumerate(local_indexes):
                # Validate local_indexes
                if idx < 0 or idx >= self.storage_size:
                    raise ValueError(
                        f"StorageUnitData put_data operation receive invalid local_index: {idx} beyond "
                        f"storage_size: {self.storage_size}"
                    )

                if f not in self.field_data:
                    # Initialize new field value list with None
                    self.field_data[f] = [None] * self.storage_size

                self.field_data[f][idx] = field_data[f][i]

    def clear(self, local_indexes: list[int]) -> None:
        """
        Clear data at specified local_indexes by setting all related fields to None.

        param:
            local_indexes: local_indexes to clear.
        """
        # Validate local_indexes
        for idx in local_indexes:
            if idx < 0 or idx >= self.storage_size:
                raise ValueError(
                    f"StorageUnitData clear operation receive invalid local_index: {idx} beyond "
                    f"storage_size: {self.storage_size}"
                )

        # Clear data at specified local_indexes
        for f in self.field_data:
            for idx in local_indexes:
                self.field_data[f][idx] = None


# TODO(hz): Abstract TransferStorage base class to facilitate support for backends such as redis
@ray.remote(num_cpus=1)
class TransferQueueStorageSimpleUnit:
    def __init__(self, storage_size: int):
        super().__init__()
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4()}"
        self.storage_size = storage_size
        self.controller_infos: dict[str, ZMQServerInfo] = {}

        self.experience_data = StorageUnitData(self.storage_size)

        self.zmq_server_info = ZMQServerInfo.create(
            role=TransferQueueRole.STORAGE,
            id=str(self.storage_unit_id),
            ip=get_node_ip_address(),
            ports={"put_get_socket": get_free_port()},
        )
        self._init_zmq_socket()

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controllers/clients:
        - controller_handshake_sockets:
            Handshake between storage unit and controllers.
        - data_status_update_sockets:
            Broadcast data update status from storage unit to controllers when handling put operation.
        - put_get_socket:
            Handle put/get requests from clients.
        """
        self.zmq_context = zmq.Context()

        self.controller_handshake_sockets: dict[str, zmq.Socket] = {}
        self.data_status_update_sockets: dict[str, zmq.Socket] = {}

        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER)
        self.put_get_socket.bind(self.zmq_server_info.to_addr("put_get_socket"))

    def register_controller_info(self, controller_infos: dict[str, ZMQServerInfo]) -> None:
        """
        Build connections between storage unit and controllers, start put/get process.

        param:
            controller_infos: Dict with controller infos.
        """
        self.controller_infos = controller_infos

        self._init_zmq_sockets_with_controller_infos()
        self._connect_to_controller()
        self._start_process_put_get()

    def _init_zmq_sockets_with_controller_infos(self) -> None:
        """Initialize ZMQ sockets between storage unit and controllers for handshake."""
        for controller_id in self.controller_infos.keys():
            self.controller_handshake_sockets[controller_id] = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_unit_id}-controller_handshake_sockets-{uuid4()}".encode(),
            )
            self.data_status_update_sockets[controller_id] = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_unit_id}-data_status_update_sockets-{uuid4()}".encode(),
            )

    def _connect_to_controller(self) -> None:
        """Connect storage unit to all controllers."""
        connected_controllers: set[str] = set()

        # Create zmq poller for handshake confirmation between controller and storage unit
        poller = zmq.Poller()

        for controller_id, controller_info in self.controller_infos.items():
            self.controller_handshake_sockets[controller_id].connect(controller_info.to_addr("handshake_socket"))
            logger.debug(
                f"[{self.zmq_server_info.id}]: Handshake connection from storage unit id #{self.zmq_server_info.id} "
                f"to controller id #{controller_id} establish successfully."
            )

            # Send handshake request to controllers
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.HANDSHAKE,
                sender_id=self.zmq_server_info.id,
                body={
                    "storage_unit_id": self.storage_unit_id,
                    "storage_size": self.storage_size,
                },
            ).serialize()

            self.controller_handshake_sockets[controller_id].send(request_msg)
            logger.debug(
                f"[{self.zmq_server_info.id}]: Send handshake request from storage unit id #{self.zmq_server_info.id} "
                f"to controller id #{controller_id} successfully."
            )

            poller.register(self.controller_handshake_sockets[controller_id], zmq.POLLIN)

        start_time = time.time()
        while (
            len(connected_controllers) < len(self.controller_infos)
            and time.time() - start_time < CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT
        ):
            socks = dict(poller.poll(POLLER_TIMEOUT_MS))

            for controller_handshake_socket in self.controller_handshake_sockets.values():
                if controller_handshake_socket in socks:
                    response_msg = ZMQMessage.deserialize(controller_handshake_socket.recv())

                    if response_msg.request_type == ZMQRequestType.HANDSHAKE_ACK:
                        connected_controllers.add(response_msg.sender_id)
                        logger.debug(
                            f"[{self.zmq_server_info.id}]: Get handshake ACK response from "
                            f"controller id #{str(response_msg.sender_id)} to storage unit id "
                            f"#{self.zmq_server_info.id} successfully."
                        )

        if len(connected_controllers) < len(self.controller_infos):
            logger.warning(
                f"[{self.zmq_server_info.id}]: Only get {len(connected_controllers)} / {len(self.controller_infos)} "
                f"successful handshake connections to controllers from storage unit id #{self.zmq_server_info.id}"
            )

    def _start_process_put_get(self) -> None:
        """Create a daemon thread and start put/get process."""
        self.process_put_get_thread = Thread(
            target=self._process_put_get, name=f"StorageUnitProcessPutGetThread-{self.zmq_server_info.id}", daemon=True
        )
        self.process_put_get_thread.start()

    def _process_put_get(self) -> None:
        """Process put_get_socket request."""
        poller = zmq.Poller()
        poller.register(self.put_get_socket, zmq.POLLIN)

        while True:
            socks = dict(poller.poll(POLLER_TIMEOUT_MS))

            if self.put_get_socket in socks:
                identity, serialized_msg = self.put_get_socket.recv_multipart()

                try:
                    request_msg = ZMQMessage.deserialize(serialized_msg)
                    operation = request_msg.request_type
                    logger.debug(f"[{self.zmq_server_info.id}]: receive operation: {operation}, message: {request_msg}")

                    if operation == ZMQRequestType.PUT_DATA:
                        response_msg = self._handle_put(request_msg)
                    elif operation == ZMQRequestType.GET_DATA:
                        response_msg = self._handle_get(request_msg)
                    elif operation == ZMQRequestType.CLEAR_DATA:
                        response_msg = self._handle_clear(request_msg)
                    else:
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.PUT_GET_OPERATION_ERROR,
                            sender_id=self.zmq_server_info.id,
                            body={
                                "message": f"Storage unit id #{self.zmq_server_info.id} "
                                f"receive invalid operation: {operation}."
                            },
                        )
                except Exception as e:
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.PUT_GET_ERROR,
                        sender_id=self.zmq_server_info.id,
                        body={
                            "message": f"Storage unit id #{self.zmq_server_info.id} occur error in processing "
                            f"put/get/clear request, detail error message: {str(e)}."
                        },
                    )

                self.put_get_socket.send_multipart([identity, response_msg.serialize()])

    def _handle_put(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle put request, add or update data into storage unit.

        param:
            data_parts: ZMQMessage from client.
        return:
            Put data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]
            local_indexes = data_parts.body["local_indexes"]
            field_data = data_parts.body["field_data"]  # field_data should be in {field_name: [real data]} format.

            self.experience_data.put_data(field_data, local_indexes)

            # After put operation finish, send a message to the client
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE, sender_id=self.zmq_server_info.id, body={}
            )

            # Gather per-tensor dtype and shape information for each field
            # global_indexes, local_indexes, and field_data correspond one-to-one
            per_tensor_dtypes: dict[int, torch.dtype] = {}
            per_tensor_shapes: dict[int, torch.Size] = {}

            # Initialize the data structure for each global index
            for global_idx in global_indexes:
                per_tensor_dtypes[global_idx] = {}
                per_tensor_shapes[global_idx] = {}

            # For each field, extract dtype and shape for each sample
            for field in field_data.keys():
                for i, data_item in enumerate(field_data[field]):
                    global_idx = global_indexes[i]
                    per_tensor_dtypes[global_idx][field] = data_item.dtype if hasattr(data_item, "dtype") else None
                    per_tensor_shapes[global_idx][field] = data_item.shape if hasattr(data_item, "shape") else None

            # Broadcast data update message to all controllers with per-tensor dtype/shape information
            self._notify_data_update(list(field_data.keys()), global_indexes, per_tensor_dtypes, per_tensor_shapes)
            return response_msg
        except Exception as e:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to put data into storage unit id "
                    f"#{self.zmq_server_info.id}, detail error message: {str(e)}"
                },
            )

    def _notify_data_update(self, fields, global_indexes, dtypes, shapes) -> None:
        """
        Broadcast data status update to all controllers.

        param:
            fields: data update related fields.
            global_indexes: data update related global_indexes.
            dtypes: per-tensor dtypes for each field, in {global_index: {field: dtype}} format.
            shapes: per-tensor shapes for each field, in {global_index: {field: shape}} format.
        """
        # Create zmq poller for notifying data update information
        poller = zmq.Poller()

        # Connect data status update socket to all controllers
        for controller_id, controller_info in self.controller_infos.items():
            data_status_update_socket = self.data_status_update_sockets[controller_id]
            data_status_update_socket.connect(controller_info.to_addr("data_status_update_socket"))
            logger.debug(
                f"[{self.zmq_server_info.id}]: Data status update connection from "
                f"storage unit id #{self.zmq_server_info.id} to "
                f"controller id #{controller_id} establish successfully."
            )

            try:
                poller.register(data_status_update_socket, zmq.POLLIN)

                request_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE,
                    sender_id=self.zmq_server_info.id,
                    body={
                        "fields": fields,
                        "global_indexes": global_indexes,
                        "dtypes": dtypes,
                        "shapes": shapes,
                    },
                ).serialize()

                data_status_update_socket.send(request_msg)
                logger.debug(
                    f"[{self.zmq_server_info.id}]: Send data status update request "
                    f"from storage unit id #{self.zmq_server_info.id} "
                    f"to controller id #{controller_id} successfully."
                )
            except Exception as e:
                request_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR,
                    sender_id=self.zmq_server_info.id,
                    body={
                        "message": f"Failed to notify data status update information from "
                        f"storage unit id #{self.zmq_server_info.id}, "
                        f"detail error message: {str(e)}"
                    },
                ).serialize()

                data_status_update_socket.send(request_msg)

        # Make sure all controllers successfully receive data status update information.
        response_controllers: set[str] = set()
        start_time = time.time()

        while (
            len(response_controllers) < len(self.controller_infos)
            and time.time() - start_time < CONTROLLER_DATA_UPDATE_RESPONSE_TIMEOUT
        ):
            socks = dict(poller.poll(POLLER_TIMEOUT_MS))

            for data_status_update_socket in self.data_status_update_sockets.values():
                if data_status_update_socket in socks:
                    response_msg = ZMQMessage.deserialize(data_status_update_socket.recv())

                    if response_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                        response_controllers.add(response_msg.sender_id)
                        logger.debug(
                            f"[{self.zmq_server_info.id}]: Get data status update ACK response "
                            f"from controller id #{response_msg.sender_id} "
                            f"to storage unit id #{self.zmq_server_info.id} successfully."
                        )

        if len(response_controllers) < len(self.controller_infos):
            logger.warning(
                f"[{self.zmq_server_info.id}]: Storage unit id #{self.zmq_server_info.id} "
                f"only get {len(response_controllers)} / {len(self.controller_infos)} "
                f"data status update ACK responses fron controllers."
            )

    def _handle_get(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle get request, return data from storage unit.

        param:
            data_parts: ZMQMessage from client.
        return:
            Get data success response ZMQMessage, containing target data.
        """
        try:
            fields = data_parts.body["fields"]
            local_indexes = data_parts.body["local_indexes"]

            result_data = self.experience_data.get_data(fields, local_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={
                    "data": result_data,
                },
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to get data from storage unit id #{self.zmq_server_info.id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def _handle_clear(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle clear request, clear data in storage unit according to given local_indexes.

        param:
            data_parts: ZMQMessage from client, including target local_indexes.
        """
        try:
            local_indexes = data_parts.body["local_indexes"]

            self.experience_data.clear(local_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={"message": f"Clear data in storage unit id #{self.zmq_server_info.id} successfully."},
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to clear data in storage unit id #{self.zmq_server_info.id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def get_zmq_server_info(self) -> ZMQServerInfo:
        return self.zmq_server_info


class AsyncTransferQueueClient:
    def __init__(
        self,
        client_id: str,
        controller_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
        storage_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        self.client_id = client_id

        self._controllers: dict[str, ZMQServerInfo] = {}
        self._storages: dict[str, ZMQServerInfo] = {}
        self._register_servers(TransferQueueRole.CONTROLLER, controller_infos)
        self._register_servers(TransferQueueRole.STORAGE, storage_infos)

    def _register_servers(
        self,
        role: TransferQueueRole,
        server_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        mapping = self._controllers if role == TransferQueueRole.CONTROLLER else self._storages

        if not isinstance(server_infos, dict):
            server_infos = {server_infos.id: server_infos}

        for info in server_infos.values():
            if not isinstance(info, ZMQServerInfo):
                raise ValueError(f"Invalid server info for {role} {id}")

            if info.id not in mapping:
                mapping[info.id] = info
                logger.info(f"[{self.client_id}]: Registered {role} server {info.id} at {info.ip}")
            else:
                logger.warning(f"[{self.client_id}]: Server {info.id} already registered, skipping")

    @staticmethod
    def dynamic_socket(target_role: TransferQueueRole, socket_name: str):
        """Decorator to auto-manage ZMQ sockets for Controller/Storage servers (create -> connect -> inject -> close).

        Args:
            target_role (TransferQueueRole): Server type to connect to. Must be one of:
                - `TransferQueueRole.CONTROLLER`
                - `TransferQueueRole.STORAGE`
            socket_name (str): Port name (from server config) to use for ZMQ connection (e.g., "data_req_port").

        Decorated Function Rules:
            1. Must be an async class method (needs `self`).
            2. `self` requires:
            - `_controllers`/`_storages`: Server registries (match `target_role`).
            - `client_id`: Unique client ID (for socket identity).
            3. Specify target server via:
            - `target_controller` (for Controller) or `target_storage` (for Storage) arg.
            - Controller role: Uses first registered server if no ID is given.
            4. Receives ZMQ socket via `socket` keyword arg (injected by decorator).
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                if target_role == TransferQueueRole.CONTROLLER:
                    servers = self._controllers
                    target = "target_controller"
                elif target_role == TransferQueueRole.STORAGE:
                    servers = self._storages
                    target = "target_storage"
                else:
                    raise ValueError("Invalid target_role, must be CONTROLLER or STORAGE")

                server_key = kwargs.get(target)
                if server_key is None:
                    for arg in args:
                        if isinstance(arg, str) and arg in servers.keys():
                            server_key = arg
                            break
                if server_key is None and target == "target_controller":
                    server_key = next(iter(servers.keys()))

                server_info = servers.get(server_key)
                if not server_info:
                    raise RuntimeError(f"Server {server_key} not found in registered {target_role} servers")

                context = zmq.asyncio.Context()
                address = f"tcp://{server_info.ip}:{server_info.ports.get(socket_name)}"
                identity = f"{self.client_id}_to_{server_info.id}_{uuid4()}".encode()
                sock = create_zmq_socket(context, zmq.DEALER, identity=identity)

                try:
                    sock.connect(address)
                    logger.info(
                        f"[{self.client_id}]: Connected to {target_role} {server_info.id} at {address} "
                        f"with identity {identity.decode()}"
                    )

                    kwargs["socket"] = sock
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"[{self.client_id}]: Error in socket operation with {target_role} {server_info.id}: {e}"
                    )
                    raise
                finally:
                    try:
                        if not sock.closed:
                            sock.setsockopt(zmq.LINGER, -1)
                            sock.close()
                        sock.close(linger=0)
                    except Exception as e:
                        logger.warning(
                            f"[{self.client_id}]: Error closing socket to {target_role} {server_info.id}: {e}"
                        )

                    context.term()

            return wrapper

        return decorator

    @dynamic_socket(target_role=TransferQueueRole.CONTROLLER, socket_name="request_handle_socket")
    async def async_get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        global_step: int,
        mode: str = "fetch",
        get_n_samples: bool = False,
        task_name: Optional[str] = None,
        target_controller: Optional[str] = None,
        socket: Optional[zmq.asyncio.Socket] = None,
    ) -> BatchMeta:
        """Asynchronously fetches data metadata via ZMQ from the target controller.

        Args:
            data_fields (list[str]): List of fields to retrieve metadata for
            batch_size (int): Processing batch size
            global_step (int): Current training/processing step
            mode (str): Data fetch mode (TODO(hz): more details to be added)
            get_n_samples (bool): TODO(hz): more details to be added
            task_name (str): Optional task name associated with the request
            target_controller (str): ID of the target controller to send the request to
            socket (zmq.asyncio.Socket): ZMQ async socket for message transmission

        Returns:
            BatchMeta: Metadata object containing data structure, sample info, etc.
        """
        assert socket is not None
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_META,
            sender_id=self.client_id,
            receiver_id=target_controller,
            body={
                "data_fields": data_fields,
                "batch_size": batch_size,
                "global_step": global_step,
                "mode": mode,
                "get_n_samples": get_n_samples,
                "task_name": task_name,
            },
        )

        try:
            await socket.send(request_msg.serialize())
            response = await socket.recv()
            response_msg = ZMQMessage.deserialize(response)
            logger.debug(
                f"[{self.client_id}]: Client get datameta response: {response_msg} from controller {target_controller}"
            )

            if response_msg.request_type == ZMQRequestType.GET_META_RESPONSE:
                metadata = response_msg.body["metadata"]
                return metadata
            else:
                raise RuntimeError(
                    f"[{self.client_id}]: Failed to get metadata from controller {target_controller}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in get_meta: {str(e)}") from e

    async def async_put(
        self,
        data: TensorDict,
        metadata: Optional[BatchMeta] = None,
        global_step: Optional[int] = None,
    ):
        """Asynchronously writes data to appropriate Storage Units based on metadata.

        If metadata isn't provided, it will be created automatically using the insert mode
        with the provided data_columns and global_step.

        Args:
            data (torch.Tensor | tensordict.TensorDict): Data to write, either a Tensor or TensorDict
            metadata (BatchMeta, optional): Optional metadata containing index and storage unit information
            global_step (int, optional): Current step (required if no metadata is provided)

        """
        if metadata is None:
            assert global_step is not None, "global_steps must be provided if metadata is not given"

            metadata = await self.async_get_meta(
                data_fields=list(data.keys()),
                batch_size=data.batch_size[0],
                global_step=global_step,
                mode="insert",
            )

        if not metadata or metadata.size == 0:
            raise ValueError("metadata cannot be none or empty")
        logger.debug(f"[{self.client_id}]: Put data with data: {data}")
        tasks = [
            self._put_to_storage(get_transfer_info(meta_group, data), target_storage=storage_id)
            for storage_id, meta_group in metadata.storage_meta_groups.items()
        ]
        await asyncio.gather(*tasks)

        logger.info(
            f"[{self.client_id}]: step {global_step} put {metadata.size} samples to storage units successfully."
        )

    @dynamic_socket(target_role=TransferQueueRole.STORAGE, socket_name="put_get_socket")
    async def _put_to_storage(self, storage_unit_data, target_storage=None, socket=None):
        """
        Send data to a specific storage unit.
        """
        global_indexes = storage_unit_data["global_indexes"]
        local_indexes = storage_unit_data["local_indexes"]
        field_data = TensorDict(
            {field: torch.stack(storage_unit_data["field_data"][field]) for field in storage_unit_data["field_data"]}
        )

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=self.client_id,
            receiver_id=target_storage,
            body={"global_indexes": global_indexes, "local_indexes": local_indexes, "field_data": field_data},
        )
        try:
            await socket.send(request_msg.serialize())
            serialized = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized)

            if response_msg.request_type != ZMQRequestType.PUT_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to put data to storage unit {target_storage}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error in put to storage unit {target_storage}: {str(e)}") from e

    @dynamic_socket(target_role=TransferQueueRole.STORAGE, socket_name="put_get_socket")
    async def _get_from_storage(self, index_data, target_storage=None, socket=None):
        global_indexes = index_data["global_indexes"]
        local_indexes = index_data["local_indexes"]
        fields = index_data["fields"]

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=self.client_id,
            receiver_id=target_storage,
            body={"local_indexes": local_indexes, "fields": fields},
        )

        try:
            await socket.send(request_msg.serialize())
            serialized = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized)
            logger.info(f"[{self.client_id}]: get data response from storage unit {target_storage}: {response_msg}")

            if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE:
                # 返回该存储单元的数据和索引信息
                su_data = response_msg.body["data"]
                return global_indexes, fields, su_data
            else:
                raise RuntimeError(
                    f"Failed to get data from storage unit {target_storage}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error getting data from storage unit {target_storage}: {str(e)}") from e

    async def async_get_data(self, metadata: BatchMeta) -> TensorDict:
        """Asynchronously fetches data via Storage Units and organizes it into a TensorDict.

        Args:
            metadata (BatchMeta): Object containing:
                - Data location info (which Storage Units hold the data)
                - `global_indexes` to determine the ordering of merged results

        Returns:
            tensordict.TensorDict with:
                - Requested data fields (e.g., "prompt_token_ids", "response_token_ids").
                - "global_indexes" key: Maps each sample to its original global index.

        Example:
            >>> returned_td = await async_get_data(metadata)
            >>> returned_td.keys()
            dict_keys(['prompt_token_ids', 'response_token_ids', 'global_indexes'])
            >>> returned_td["prompt_token_ids"].shape  # Batch size 4, seq length 128
            torch.Size([4, 128])
            >>> returned_td["global_indexes"]  # Preserves original global order
            tensor([7, 4, 6, 5])

        Note:
            Why track `global_indexes`?
            - Batches may be rearranged during task processing. `global_indexes` retains the original
            mapping to Storage Units, enabling correct data writing back to Storage Units later.

        """
        if not metadata or metadata.size == 0:
            return TensorDict({}, batch_size=0)

        # Use optimized retrieval with direct storage group access
        tasks = [
            self._get_from_storage(meta_group.get_transfer_info(), target_storage=storage_id)
            for storage_id, meta_group in metadata.storage_meta_groups.items()
        ]

        results = await asyncio.gather(*tasks)

        # global_index: {field1: value, field2: value, ...}
        storage_data: dict[int, dict[str, torch.Tensor]] = {}
        for global_indexes, fields, storage_unit_data in results:
            for idx, global_idx in enumerate(global_indexes):
                if global_idx not in storage_data:
                    storage_data[global_idx] = {}
                for field in fields:
                    storage_data[global_idx][field] = storage_unit_data[field][idx]

        ordered_data: dict[str, torch.Tensor] = {field: [] for field in metadata.fields}
        for global_idx in metadata.global_indexes:
            for field in metadata.fields:
                ordered_data[field].append(storage_data[global_idx][field])

        tensor_data = {field: torch.stack(v) for field, v in ordered_data.items()}
        tensor_data["global_indexes"] = torch.tensor(metadata.global_indexes)

        return TensorDict(tensor_data, batch_size=len(storage_data))

    async def async_clear(self, global_step: int):
        """Asynchronously clears data from all storage units and controller metadata.

        Args:
            global_step (int): The training step associated with the clear operation

        """
        try:
            target_controller = next(iter(self._controllers.keys()))
            metadata = await self._get_clear_meta(global_step, target_controller)

            tasks = []

            for target_controller in self._controllers.keys():
                tasks.append(self._clear_controller(global_step, target_controller))

            # Group samples by storage unit for clearing
            for target_storage, group in metadata.storage_meta_groups.items():
                group_info = group.get_transfer_info()
                if target_storage not in self._storages:
                    logger.warning(
                        f"[{self.client_id}]: Storage unit {target_storage} not registered, skipping clear operation."
                    )
                    continue
                tasks.append(
                    self._clear_storage_unit(
                        group_info["local_indexes"],
                        target_storage,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[{self.client_id}]: Error in clear operation task {i}: {result}")

            logger.info(f"[{self.client_id}]: Clear operation for global_step {global_step} completed.")
        except Exception as e:
            raise RuntimeError(f"Error in clear operation: {str(e)}") from e

    @dynamic_socket(target_role=TransferQueueRole.CONTROLLER, socket_name="request_handle_socket")
    async def _get_clear_meta(self, global_step: int, target_controller=None, socket=None):
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_CLEAR_META,
            sender_id=self.client_id,
            receiver_id=target_controller,
            body={"global_step": global_step},
        )

        await socket.send(request_msg.serialize())
        serialized = await socket.recv()
        response_msg = ZMQMessage.deserialize(serialized)

        if response_msg.request_type != ZMQRequestType.GET_CLEAR_META_RESPONSE:
            raise RuntimeError(
                f"Failed to get metadata for clear operation: {response_msg.body.get('message', 'Unknown error')}"
            )

        return response_msg.body["metadata"]

    @dynamic_socket(target_role=TransferQueueRole.CONTROLLER, socket_name="request_handle_socket")
    async def _clear_controller(self, global_step, target_controller=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_META,
                sender_id=self.client_id,
                receiver_id=target_controller,
                body={"global_step": global_step},
            )

            await socket.send(request_msg.serialize())
            serialized_msg = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_msg)

            if response_msg.request_type != ZMQRequestType.CLEAR_META_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear controller {target_controller}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(
                f"[{self.client_id}]: Successfully clear controller {target_controller} for global_step {global_step}"
            )
        except Exception as e:
            logger.error(f"[{self.client_id}]: Error clearing controller {target_controller}: {str(e)}")
            raise

    @dynamic_socket(target_role=TransferQueueRole.STORAGE, socket_name="put_get_socket")
    async def _clear_storage_unit(self, local_indexes, target_storage=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA,
                sender_id=self.client_id,
                receiver_id=target_storage,
                body={"local_indexes": local_indexes},
            )

            await socket.send(request_msg.serialize())
            serialized_msg = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_msg)

            if response_msg.request_type != ZMQRequestType.CLEAR_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear storage {target_storage}: {response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(f"[{self.client_id}]: Successfully clear storage unit {target_storage}")
        except Exception as e:
            logger.error(f"[{self.client_id}]: Error clearing storage unit {target_storage}: {str(e)}")
            raise


class TransferQueueClient(AsyncTransferQueueClient):
    def __init__(
        self,
        client_id: str,
        controller_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
        storage_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        super().__init__(
            client_id,
            controller_infos,
            storage_infos,
        )

    def put(self, data: TensorDict, metadata: Optional[BatchMeta] = None, global_step: Optional[int] = None):
        return asyncio.run(self.async_put(data, metadata, global_step))

    def get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        global_step: int,
        get_n_samples: bool = False,
        task_name: Optional[str] = None,
    ) -> BatchMeta:
        return asyncio.run(
            self.async_get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                global_step=global_step,
                get_n_samples=get_n_samples,
                task_name=task_name,
            )
        )

    def get_data(self, metadata: BatchMeta) -> TensorDict:
        return asyncio.run(self.async_get_data(metadata))

    def clear(self, global_step: int):
        return asyncio.run(self.async_clear(global_step))

    def check_current_step_consumption(self, global_step):
        # 检查当前global batch是否消耗完
        # TODO: Implement step consumption check
        pass


def process_zmq_server_info(handlers: dict[Any, Union[TransferQueueController, TransferQueueStorageSimpleUnit]]):  # noqa: UP007
    server_info = {}
    for name, handler in handlers.items():
        server_info[name] = ray.get(handler.get_zmq_server_info.remote())  # type: ignore[attr-defined]
    return server_info


def _add_field_data(
    transfer_dict: dict[str, Any], storage_meta_group: StorageMetaGroup, data: TensorDict
) -> dict[str, Any]:
    """Helper function to add field data to the transfer dictionary"""
    field_names = transfer_dict["fields"]
    for fname in field_names:
        if fname in data.keys():
            transfer_dict["field_data"][fname] = []
            for sample_meta in storage_meta_group.sample_metas:
                transfer_dict["field_data"][fname].append(data[fname][sample_meta.batch_index])
    return transfer_dict


def get_transfer_info(
    storage_meta_group: StorageMetaGroup,
    data: TensorDict,
) -> dict[str, Any]:
    """Convert to dictionary format with field data for put operations"""
    result = storage_meta_group.get_transfer_info(field_names=data.keys())
    result = _add_field_data(result, storage_meta_group, data)
    return result


def _union_fields(fields1: dict[str, FieldMeta], fields2: dict[str, FieldMeta]) -> TensorDict:
    """Union two sample's fields."""
    for name in fields2.keys():
        if name not in fields1:
            fields1[name] = fields2[name]
        else:
            assert fields1[name].equals(fields2[name]), f"{name} in fields1 and fields2 are not the same object"

    return fields1
