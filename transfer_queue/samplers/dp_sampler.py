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

from typing import Any

import torch

from transfer_queue.samplers.base import BaseSampler


class DPSampler(BaseSampler):
    """Data Parallel (DP) aware sampler.

    This sampler ensures that:
    1. Ranks within the same DP group get the same data
    2. Different DP groups get different data

    Use case: In distributed training with data parallelism, replicas within
    a DP group need identical data for gradient synchronization, while different
    DP groups should process different data for efficiency.

    Args:
        dp_rank: Rank of this process within the DP group (0-indexed)
        dp_size: Total number of DP groups
        world_rank: Global rank of this process
        world_size: Total number of processes
    """

    def __init__(
        self,
        dp_rank: int,
        dp_size: int,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize DP sampler.

        Args:
            dp_rank: Rank within the DP group (0 to dp_size-1)
            dp_size: Number of DP groups
            world_rank: Global rank of this process (for logging/debugging)
            world_size: Total number of processes (for validation)

        Raises:
            ValueError: If dp_rank or dp_size are invalid
        """
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(
                f"dp_rank ({dp_rank}) must be in range [0, {dp_size})"
            )
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.world_rank = world_rank
        self.world_size = world_size

    def sample(
        self,
        ready_indices: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> list[int]:
        """Sample data for the current DP group.

        The sampler partitions ready_indices across DP groups and selects
        data specific to this DP group.

        Args:
            ready_indices: List of global indices ready for consumption
            batch_size: Number of samples to select for this DP group
            **kwargs: Unused for DP sampling

        Returns:
            Indices assigned to this DP group

        Raises:
            ValueError: If insufficient data is available for the DP configuration
        """
        total_required = batch_size * self.dp_size

        if len(ready_indices) < total_required:
            raise ValueError(
                f"Insufficient ready indices for DP sampling: "
                f"requested {batch_size} per DP group × {self.dp_size} groups = {total_required}, "
                f"available {len(ready_indices)}"
            )

        # Partition the ready indices across DP groups
        # Each DP group gets batch_size consecutive samples
        start_idx = self.dp_rank * batch_size
        end_idx = start_idx + batch_size

        return ready_indices[start_idx:end_idx]

    def filter_ready_indices(
        self,
        all_indices: list[int],
        production_status: torch.Tensor,
        consumption_status: torch.Tensor,
        data_fields: list[str],
        field_mapping: dict[str, int],
        **kwargs: Any,
    ) -> list[int]:
        """Filter indices using standard readiness check.

        For DP sampling, the readiness check is the same as sequential sampling.
        The DP-specific logic is applied in the sample() method.

        Args:
            all_indices: List of candidate indices
            production_status: Tensor of shape [total_storage_size, num_fields]
            consumption_status: Tensor of shape [total_storage_size]
            data_fields: List of required field names
            field_mapping: Mapping from field names to column indices
            **kwargs: Unused for DP sampling

        Returns:
            List of indices that meet readiness criteria
        """
        if not all_indices:
            return []

        # Get column indices for required fields
        col_indices = [field_mapping[field] for field in data_fields if field in field_mapping]
        if not col_indices:
            return []

        # Check if all required fields are ready for each index
        ready_mask = torch.zeros(len(all_indices), dtype=torch.bool)
        for i, idx in enumerate(all_indices):
            all_fields_ready = torch.all(production_status[idx, col_indices] == 1).item()
            not_consumed = consumption_status[idx] == 0
            ready_mask[i] = all_fields_ready and not_consumed

        # Return indices where ready_mask is True
        ready_indices = [all_indices[i] for i in range(len(all_indices)) if ready_mask[i]]
        return ready_indices
