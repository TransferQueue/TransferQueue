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


class SequentialSampler(BaseSampler):
    """Sequential sampler that maintains the original TransferQueue behavior.

    This sampler selects samples in sequential order from the ready indices.
    It is the default sampler and preserves backward compatibility.
    """

    def sample(
        self,
        ready_indices: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> list[int]:
        """Sample indices sequentially from ready_indices.

        Args:
            ready_indices: List of global indices ready for consumption
            batch_size: Number of samples to select
            **kwargs: Unused for sequential sampling

        Returns:
            First batch_size indices from ready_indices

        Raises:
            ValueError: If batch_size exceeds available ready indices
        """
        if len(ready_indices) < batch_size:
            raise ValueError(
                f"Insufficient ready indices: requested {batch_size}, "
                f"available {len(ready_indices)}"
            )

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
        """Filter indices using standard readiness check.

        An index is ready if:
        1. All required fields are produced (production_status == 1)
        2. The sample has not been consumed (consumption_status == 0)

        Args:
            all_indices: List of candidate indices
            production_status: Tensor of shape [total_storage_size, num_fields]
            consumption_status: Tensor of shape [total_storage_size]
            data_fields: List of required field names
            field_mapping: Mapping from field names to column indices
            **kwargs: Unused for sequential sampling

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
            # Check if all fields are produced and sample not consumed
            all_fields_ready = torch.all(production_status[idx, col_indices] == 1).item()
            not_consumed = consumption_status[idx] == 0
            ready_mask[i] = all_fields_ready and not_consumed

        # Return indices where ready_mask is True
        ready_indices = [all_indices[i] for i in range(len(all_indices)) if ready_mask[i]]
        return ready_indices
