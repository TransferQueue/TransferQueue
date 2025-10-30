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

import dataclasses
import itertools
from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from tensordict import TensorDict

from transfer_queue.utils.utils import ProductionStatus


@dataclass
class FieldMeta:
    """Records the metadata of a single data field (name, dtype, shape, etc.)."""

    name: str
    dtype: Optional[Any]  # Data type (e.g., torch.float32, numpy.float32)
    shape: Optional[Any]  # Data shape (e.g., torch.Size([3, 224, 224]), (3, 224, 224))
    production_status: ProductionStatus = ProductionStatus.NOT_PRODUCED

    def __str__(self) -> str:
        return (
            f"FieldMeta(name='{self.name}', dtype={self.dtype}, "
            f"shape={self.shape}, production_status={self.production_status})"
        )

    @property
    def is_ready(self) -> bool:
        """Check if this field is ready for consumption"""
        return self.production_status == ProductionStatus.READY_FOR_CONSUME


@dataclass
class SampleMeta:
    """Records the metadata of a single data sample (stored as a row in the data system)."""

    global_step: int  # Global step, used for data versioning
    global_index: int  # Global row index, uniquely identifies a data sample
    fields: dict[str, FieldMeta]  # Fields of interest for this sample

    def __post_init__(self):
        """Initialize is_ready property based on field readiness"""
        # Check if all fields are ready and update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))

    def __str__(self) -> str:
        return f"SampleMeta(global_step={self.global_step}, global_index={self.global_index})"

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

    def add_fields(self, fields: dict[str, FieldMeta]) -> "SampleMeta":
        """
        Add new fields to this sample. New fields will be initialized with given dtype, shape
        and production_status (if provided). If not provided, default values (None, None, READY_FOR_CONSUME)
        will be used. This modifies the sample in-place to include the new fields.
        """
        self.fields = _union_fields(self.fields, fields)
        # Update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))
        return self

    def union(self, other: "SampleMeta", validate: bool = True) -> "SampleMeta":
        """
        Create a union of this sample's fields with another sample's fields.
        Assume both samples have the same global index. If fields overlap, the
        fields in this sample will be replaced by the other sample's fields.

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
        self.fields = _union_fields(self.fields, other.fields)

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
class BatchMeta:
    """Records the metadata of a batch of data samples."""

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

            # assume all samples have the same fields.
            object.__setattr__(self, "_field_names", sorted(self.samples[0].field_names))
        else:
            object.__setattr__(self, "_global_indexes", [])
            object.__setattr__(self, "_field_names", [])

    @property
    def size(self) -> int:
        """Return the number of samples in this batch"""
        return getattr(self, "_size", 0)

    @property
    def global_indexes(self) -> list[int]:
        """Get all global indexes in this batch"""
        return getattr(self, "_global_indexes", [])

    @property
    def field_names(self) -> list[str]:
        """Get all unique field names in this batch"""
        return getattr(self, "_field_names", [])

    @property
    def is_ready(self) -> bool:
        """Check if all samples in this batch are ready for consumption"""
        # TODO: get ready status from controller realtime
        return getattr(self, "_is_ready", False)

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

    def add_fields(self, tensor_dict: TensorDict, set_all_ready: bool = True) -> "BatchMeta":
        """
        Add new fields from a TensorDict to all samples in this batch.
        This modifies each sample in-place to include the new fields.

        Args:
            tensor_dict (TensorDict): The input TensorDict containing new fields.
            set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME. Default is True.
        """
        fields = _extract_field_metas(tensor_dict, set_all_ready)

        if fields:
            if len(self.samples) != len(fields):
                raise ValueError(f"add_fields length mismatch: samples={len(self.samples)} vs fields={len(fields)}")
        for idx, sample in enumerate(self.samples):
            sample.add_fields(fields=fields[idx])

            # Update batch-level fields cache
            object.__setattr__(self, "_field_names", sorted(self.samples[0].field_names))
            object.__setattr__(self, "_is_ready", all(sample.is_ready for sample in self.samples))
        return self

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
            chunk = BatchMeta(samples=chunk_samples, extra_info=self.extra_info.copy())
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
            Concatenated BatchMeta

        Raises:
            ValueError: If validation fails (e.g., field names do not match)
        """
        if not data:
            return None

        if validate:
            base_fields = data[0].field_names

            for chunk in data:
                if chunk.field_names != base_fields:
                    raise ValueError("Error: Field names do not match for concatenation.")

        # Combine all samples
        all_samples = list(itertools.chain.from_iterable(chunk.samples for chunk in data))
        # Merge all extra_info dictionaries from the chunks
        merged_extra_info = dict(ChainMap(*(chunk.extra_info for chunk in data)))

        return BatchMeta(samples=all_samples, extra_info=merged_extra_info)

    def union(self, other: "BatchMeta", validate: bool = True) -> Optional["BatchMeta"]:
        """
        Create a union of this batch's fields with another batch's fields.
        Assume both batches have the same global indices. If fields overlap, the
        fields in this batch will be replaced by the other batch's fields.

        Args:
            other: Another BatchMeta to union with
            validate: Whether to validate union conditions

        Returns:
            New BatchMeta with unioned fields

        Raises:
            ValueError: If validation fails (e.g., batch sizes or global indexes do not match)
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

    def reorder(self, indices: list[int]):
        """
        Reorder the SampleMeta in the BatchMeta according to the given indices.

        The operation is performed in-place, modifying the current BatchMeta's SampleMeta order.

        Args:
            indices : list[int]
                A list of integers specifying the new order of SampleMeta. Each integer
                represents the current index of the SampleMeta in the BatchMeta.
        """
        # Reorder the samples
        reordered_samples = [self.samples[i] for i in indices]
        object.__setattr__(self, "samples", reordered_samples)

        # Update necessary attributes
        self._update_after_reorder()

    def _update_after_reorder(self) -> None:
        """Update related attributes specifically for the reorder operation"""
        # Update batch_index for each sample
        for idx, sample in enumerate(self.samples):
            object.__setattr__(sample, "_batch_index", idx)

        # Update cached index lists
        object.__setattr__(self, "_global_indexes", [sample.global_index for sample in self.samples])

        # Note: No need to update _size, _field_names, _is_ready, etc., as these remain unchanged after reorder

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


def _union_fields(fields1: dict[str, FieldMeta], fields2: dict[str, FieldMeta]) -> dict[str, FieldMeta]:
    """Union two sample's fields. If fields overlap, the fields in fields1 will be replaced by fields2."""
    for name in fields2.keys():
        fields1[name] = fields2[name]
    return fields1


def _extract_field_metas(tensor_dict: TensorDict, set_all_ready: bool = True) -> list[dict[str, FieldMeta]]:
    """
    Extract field metas from a TensorDict. If data in tensor_dict does not have dtype or shape attribute,
    the corresponding dtype or shape will be set to None.

    Args:
        tensor_dict (TensorDict): The input TensorDict.
        set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME.
                              Otherwise, set to NOT_PRODUCED. Default is True.

    Returns:
        all_fields (list[dict[str, FieldMeta]]): A list of dictionaries containing field metadata.
    """
    batch_size = tensor_dict.batch_size[0]

    production_status = ProductionStatus.READY_FOR_CONSUME if set_all_ready else ProductionStatus.NOT_PRODUCED

    all_fields = [
        {
            name: FieldMeta(
                name=name,
                dtype=getattr(value, "dtype", None),
                shape=getattr(value, "shape", None),
                production_status=production_status,
            )
            for name, value in tensor_dict[idx].items()
        }
        for idx in range(batch_size)
    ]

    return all_fields
