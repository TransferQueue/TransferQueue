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
from typing import Any, Callable, Optional

import torch

from transfer_queue.samplers.base import BaseSampler

logger = logging.getLogger(__name__)


class DPSampler(BaseSampler):
    """Data Parallel (DP) aware stateful sampler.

    This sampler ensures that:
    1. Ranks within the same DP group get the same data
    2. Different DP groups get different data
    3. Data is marked as consumed only after all ranks in a DP group have consumed it

    Use case: In distributed training with data parallelism, replicas within
    a DP group need identical data for gradient synchronization, while different
    DP groups should process different data for efficiency.

    This is a stateful sampler that maintains state across multiple requests
    to ensure coordinated data access within DP groups.

    Args:
        dp_rank: Rank of this process within the DP group (0-indexed)
        dp_size: Total number of DP groups
        world_rank: Global rank of this process
        world_size: Total number of processes
    """

    def __init__(
        self,
        dp_rank: int = 0,
        dp_size: int = 1,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Initialize DP sampler.

        Args:
            dp_rank: Rank within the DP group (0 to dp_size-1). Can be overridden via sampler_params.
            dp_size: Number of DP groups. Can be overridden via sampler_params.
            world_rank: Global rank of this process (for logging/debugging)
            world_size: Total number of processes (for validation)

        Raises:
            ValueError: If dp_rank or dp_size are invalid
        """
        super().__init__()  # Initialize base sampler's consumption tracking
        
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
        
        # DP-specific state: {task_name: {group_id: (cached_indices, consumed_ranks)}}
        # This tracks which DP group is consuming which data and which ranks have consumed
        self._group_state: dict[str, dict[str, tuple[list[int], set[int]]]] = {}

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

    def is_stateful(self) -> bool:
        """DPSampler is stateful to coordinate data access across ranks.
        
        Returns:
            True, indicating this sampler maintains state
        """
        return True

    def get_or_create_batch(
        self,
        task_name: str,
        sampler_params: dict[str, Any],
        scan_ready_fn: Callable[[], list[int]],
        **kwargs: Any,
    ) -> tuple[list[int], Optional[list[int]]]:
        """Get or create a batch for DP group with state management.
        
        This method implements stateful DP sampling where:
        1. The first rank in a DP group to request data triggers batch creation
        2. Subsequent ranks in the same group get the cached batch
        3. Data is marked consumed only after all ranks have retrieved it
        
        The sampler maintains its own consumption state to track which DP group
        consumed which data.
        
        Args:
            task_name: Name of the consumer task
            sampler_params: Must contain 'dp_rank', 'dp_size', and 'dp_group_id'
            scan_ready_fn: Function to scan for ready indices (production status checked)
            **kwargs: Must contain 'batch_size', 'data_fields', 'global_step', 'get_n_samples'
        
        Returns:
            Tuple of (batch_indices, indices_to_mark_consumed)
            - batch_indices: Indices for this specific rank
            - indices_to_mark_consumed: All indices to mark consumed, or None if not all ranks done
        """
        # Extract parameters
        dp_rank = sampler_params.get("dp_rank", self.dp_rank)
        dp_size = sampler_params.get("dp_size", self.dp_size)
        dp_group_id = sampler_params.get("dp_group_id")
        batch_size = kwargs.get("batch_size")
        
        if dp_group_id is None:
            raise ValueError("dp_group_id is required for stateful DP sampling")
        if batch_size is None:
            raise ValueError("batch_size is required")
        
        # Initialize task state if needed
        if task_name not in self._group_state:
            self._group_state[task_name] = {}
        
        # Check if this DP group already has cached data
        if dp_group_id in self._group_state[task_name]:
            cached_indices, consumed_ranks = self._group_state[task_name][dp_group_id]
            
            # Mark this rank as having consumed the data
            consumed_ranks.add(dp_rank)
            
            # Return the appropriate slice for this DP rank
            start_idx = (dp_rank % dp_size) * batch_size
            end_idx = start_idx + batch_size
            rank_indices = cached_indices[start_idx:end_idx]
            
            # Check if all ranks have consumed
            if len(consumed_ranks) >= dp_size:
                # All ranks have consumed, cleanup DP state and mark consumed
                del self._group_state[task_name][dp_group_id]
                
                # Track in sampler's consumption state
                self.mark_consumed(task_name, cached_indices)
                
                logger.info(
                    f"All {dp_size} ranks consumed data for DP group {dp_group_id}, "
                    f"marking {len(cached_indices)} indices as consumed"
                )
                # Return all indices to mark as consumed
                return rank_indices, cached_indices
            
            # Not all ranks have consumed yet, don't mark as consumed
            return rank_indices, None
        
        # This is the first rank in the DP group to request data
        # Scan for ready data (filtering out already consumed indices)
        ready_for_consume_idx = scan_ready_fn()
        
        # Filter out indices already consumed by this sampler
        ready_for_consume_idx = [
            idx for idx in ready_for_consume_idx 
            if not self.is_consumed(task_name, idx)
        ]
        
        # Need enough data for all DP groups
        total_required = batch_size * dp_size
        if len(ready_for_consume_idx) < total_required:
            raise ValueError(
                f"Insufficient ready indices for DP sampling: "
                f"required {total_required} ({batch_size} × {dp_size} groups), "
                f"available {len(ready_for_consume_idx)}"
            )
        
        # Get data for all DP groups (not just this rank)
        # Simply take the first total_required indices sequentially
        all_indices_for_group = ready_for_consume_idx[:total_required]
        
        # Cache the data for this DP group
        consumed_ranks = {dp_rank}  # This rank has now consumed
        self._group_state[task_name][dp_group_id] = (all_indices_for_group, consumed_ranks)
        
        logger.info(
            f"Created DP batch for group {dp_group_id}: {len(all_indices_for_group)} total indices, "
            f"rank {dp_rank} is first consumer"
        )
        
        # Return the appropriate slice for this DP rank
        start_idx = (dp_rank % dp_size) * batch_size
        end_idx = start_idx + batch_size
        return all_indices_for_group[start_idx:end_idx], None
