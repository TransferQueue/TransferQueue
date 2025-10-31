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

import torch

from transfer_queue.sampler import BaseSampler


class GRPOGroupNSampler(BaseSampler):
    def __init__(
        self,
    ):
        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        n_samples_per_prompt: int,
    ) -> tuple[list[int], list[int]]:
        assert batch_size % n_samples_per_prompt == 0
        batch_size_n_samples = batch_size // n_samples_per_prompt

        group_ready_for_consume_idx = torch.tensor(ready_indexes, dtype=torch.int).view(-1, n_samples_per_prompt)

        sampled_indexes = group_ready_for_consume_idx[list(range(batch_size_n_samples))].flatten().tolist()
        consumed_indexes = sampled_indexes

        return sampled_indexes, consumed_indexes

        sampled_indices = ready_indices[:batch_size]
        consumed_indices = sampled_indices

        return consumed_indices, sampled_indices
