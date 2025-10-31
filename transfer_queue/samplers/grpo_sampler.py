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


class GRPOSampler(BaseSampler):
    """GRPO (Group Relative Policy Optimization) sampler.

    This sampler requires that n_samples for each prompt are ready simultaneously
    before they can be consumed. It groups samples by prompts and only returns
    groups where all n_samples are ready.

    Use case: In GRPO, multiple responses are generated for each prompt and need
    to be processed together for computing relative advantages.

    Args:
        num_n_samples: Number of samples per prompt (group size)
    """

    def __init__(self, num_n_samples: int):
        """Initialize GRPO sampler.

        Args:
            num_n_samples: Number of samples per prompt that must be ready together
        """
        super().__init__()
        if num_n_samples < 1:
            raise ValueError(f"num_n_samples must be >= 1, got {num_n_samples}")
        self.num_n_samples = num_n_samples

    def sample(
        self,
        ready_indices: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> list[int]:
        """Sample complete groups sequentially from ready indices.

        Args:
            ready_indices: List of global indices ready for consumption, expected to
                          be organized in groups of num_n_samples
            batch_size: Total number of samples to select (must be divisible by num_n_samples)
            **kwargs: Unused for GRPO sampling

        Returns:
            Sequentially selected indices from complete groups

        Raises:
            ValueError: If batch_size is not divisible by num_n_samples or
                       if insufficient complete groups are available
        """
        if batch_size % self.num_n_samples != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"num_n_samples ({self.num_n_samples})"
            )

        if len(ready_indices) < batch_size:
            raise ValueError(
                f"Insufficient ready indices: requested {batch_size}, "
                f"available {len(ready_indices)}"
            )

        # Verify that ready_indices are properly grouped
        if len(ready_indices) % self.num_n_samples != 0:
            raise ValueError(
                f"Ready indices length ({len(ready_indices)}) must be divisible by "
                f"num_n_samples ({self.num_n_samples})"
            )

        # Take the first batch_size samples (already grouped)
        return ready_indices[:batch_size]

    def filter_ready_indices(
        self,
        all_indices: list[int],
        production_status: torch.Tensor,
        consumption_status: torch.Tensor,
        data_fields: list[str],
        field_mapping: dict[str, int],
        **kwargs: Any,
    ) -> list[int]:
        """Filter indices to return only complete groups where all n_samples are ready.

        For GRPO, we need to ensure that for each prompt, all n_samples responses
        are ready before any can be consumed.

        Args:
            all_indices: List of candidate indices
            production_status: Tensor of shape [total_storage_size, num_fields]
            consumption_status: Tensor of shape [total_storage_size]
            data_fields: List of required field names
            field_mapping: Mapping from field names to column indices
            **kwargs: Unused for GRPO sampling

        Returns:
            List of indices from complete groups where all samples in the group are ready
        """
        if not all_indices:
            return []

        # Ensure indices are grouped correctly
        if len(all_indices) % self.num_n_samples != 0:
            # Truncate to the largest multiple of num_n_samples
            truncated_size = (len(all_indices) // self.num_n_samples) * self.num_n_samples
            all_indices = all_indices[:truncated_size]

        if not all_indices:
            return []

        # Get column indices for required fields
        col_indices = [field_mapping[field] for field in data_fields if field in field_mapping]
        if not col_indices:
            return []

        # Create a mask for individual sample readiness
        ready_mask = torch.zeros(len(all_indices), dtype=torch.bool)
        for i, idx in enumerate(all_indices):
            all_fields_ready = torch.all(production_status[idx, col_indices] == 1).item()
            not_consumed = consumption_status[idx] == 0
            ready_mask[i] = all_fields_ready and not_consumed

        # Reshape to group view and check if entire groups are ready
        ready_mask_grouped = ready_mask.view(-1, self.num_n_samples)
        groups_ready = torch.all(ready_mask_grouped, dim=1)

        # Get indices of ready groups
        ready_group_indices = groups_ready.nonzero(as_tuple=False).flatten()

        # Calculate all sample indices from ready groups
        sample_offset = torch.arange(self.num_n_samples)
        ready_indices_flat = (
            (ready_group_indices.unsqueeze(1) * self.num_n_samples + sample_offset)
            .flatten()
            .tolist()
        )

        # Map back to original global indices
        ready_indices = [all_indices[i] for i in ready_indices_flat]
        return ready_indices
