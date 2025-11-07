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

from abc import ABC, abstractmethod
from typing import Any


class BaseSampler(ABC):
    """Base class for samplers that control how data is consumed from TransferQueue.

    A sampler defines the logic for selecting which samples to retrieve from the
    available samples, and which should be labeled as consumed (will never be retrieved in the future).
    Based on this abstraction, users can implement load-balancing strategies for better performance,
    or define their own data reconsumption strategies.

    The sampler interface provides a clean separation between data availability management
    (handled by TransferQueueController) and data consumption strategy (implemented by samplers).
    This allows users to customize sampling behavior without modifying the core queue logic.

    Available Samplers:
    - **SequentialSampler**: Default sampler, selects samples sequentially
    - **GRPOGroupNSampler**: Groups samples for reinforcement learning workflows
    - **RankAwareSampler**: Rank-aware sampling for distributed scenarios (TODO)
    - **Custom Samplers**: Users can extend BaseSampler for domain-specific needs

    Integration Flow:
    1. Client requests metadata via `get_meta(sampling_config=...)`
    2. Controller collects ready_indexes from storage units
    3. Controller calls sampler.sample(ready_indexes, batch_size, **sampling_config)
    4. Sampler returns (sampled_indexes, consumed_indexes)
    5. Controller marks consumed_indexes as consumed in metadata
    6. Client retrieves data for sampled_indexes

    Implementation Guidelines:
    - Always return both sampled and consumed indexes (may be identical)
    - Validate input parameters appropriately
    - Handle edge cases gracefully (empty lists, zero batch sizes)
    - Document any sampler-specific configuration parameters
    - Consider performance for large-scale workloads
    """

    def __init__(self):
        self._states: dict[str, Any] = {}

    @abstractmethod
    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample a batch of indices from the ready indices.

        Args:
            ready_indexes: List of global indices for which all required fields of the
            corresponding samples have been produced, and the samples are not labeled as
            consumed in the corresponding task.
            batch_size: Number of samples to select
            *args: Additional positional arguments for specific sampler implementations
            **kwargs: Additional keyword arguments for specific sampler implementations

        Returns:
            List of sampled global indices of length batch_size

            List of global indices of length batch_size that should be labeled as consumed
            (will never be retrieved in the future)

        Raises:
            ValueError: If batch_size is invalid or ready_indexes is insufficient
        """
        raise NotImplementedError("Subclasses must implement sample")

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[list[int], list[int]]:
        return self.sample(*args, **kwargs)
