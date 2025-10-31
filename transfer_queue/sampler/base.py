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
    available samples. Based on this abstraction, users can implement load-balancing
    strategies for better performance, or define their own data reconsumption strategies.

    We provide sequential sampling (SequentialSampler), grouped sampling (GRPOGroupNSampler),
    and distributed sampling (DistributedSampler) as reference implementation.
    """

    def __init__(self):
        self._states: dict[str, Any] = None

    @abstractmethod
    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample a batch of indices from the ready indices.

        Args:
            ready_indexes: List of global indices that are ready for consumption
            batch_size: Number of samples to select
            **kwargs: Additional sampler-specific parameters

        Returns:
            List of sampled global indices of length batch_size
            List of global indices of length batch_size that should be labeled as consumed

        Raises:
            ValueError: If batch_size is invalid or ready_indices is insufficient
        """
        raise NotImplementedError("Subclasses must implement sample")

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[list[int], list[int]]:
        return self.sample(*args, **kwargs)
