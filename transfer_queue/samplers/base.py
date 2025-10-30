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
    available ready samples. This abstraction allows different consumption patterns
    such as sequential sampling, grouped sampling (GRPO), or distributed sampling (DP).

    Similar to torchrl.data.ReplayBuffer's sampler interface, samplers can be
    passed as either instances or callables that construct sampler instances.
    """

    @abstractmethod
    def sample(
        self,
        ready_indices: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> list[int]:
        """Sample a batch of indices from the ready indices.

        Args:
            ready_indices: List of global indices that are ready for consumption
            batch_size: Number of samples to select
            **kwargs: Additional sampler-specific parameters

        Returns:
            List of sampled global indices of length batch_size

        Raises:
            ValueError: If batch_size is invalid or ready_indices is insufficient
        """
        pass

    @abstractmethod
    def filter_ready_indices(
        self,
        all_indices: list[int],
        production_status: Any,
        consumption_status: Any,
        data_fields: list[str],
        field_mapping: dict[str, int],
        **kwargs: Any,
    ) -> list[int]:
        """Filter indices to find those that are ready for consumption.

        This method allows samplers to define custom logic for determining which
        samples are "ready" based on the production and consumption status.

        Args:
            all_indices: List of all candidate indices in the current step
            production_status: Tensor tracking production status of all fields
            consumption_status: Tensor tracking consumption status for this task
            data_fields: List of required field names
            field_mapping: Mapping from field names to column indices
            **kwargs: Additional sampler-specific parameters

        Returns:
            List of indices that are ready for consumption according to sampler logic
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> list[int]:
        """Allow samplers to be called directly."""
        return self.sample(*args, **kwargs)
