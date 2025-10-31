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
from typing import Any, Callable, Optional


class BaseSampler(ABC):
    """Base class for samplers that control how data is consumed from TransferQueue.

    A sampler defines the logic for selecting which samples to retrieve from the
    available ready samples. This abstraction allows different consumption patterns
    such as sequential sampling, grouped sampling (GRPO), or distributed sampling (DP).

    Similar to torchrl.data.ReplayBuffer's sampler interface, samplers can be
    passed as either instances or callables that construct sampler instances.
    
    Samplers can be stateless or stateful:
    - Stateless samplers (e.g., SequentialSampler, GRPOSampler) operate independently on each request
    - Stateful samplers (e.g., DPSampler) maintain state across multiple requests and handle
      coordinated data access patterns, including their own consumption tracking
    """
    
    def __init__(self):
        """Initialize the base sampler.
        
        Subclasses should call super().__init__() to initialize consumption tracking.
        """
        # Consumption state managed by sampler: {task_name: set of consumed indices}
        self._consumption_state: dict[str, set[int]] = {}

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

    def is_stateful(self) -> bool:
        """Check if this sampler requires state management.
        
        Stateful samplers need to coordinate data access across multiple processes
        or requests. The controller will delegate state management to the sampler.
        
        Returns:
            True if the sampler is stateful, False otherwise (default)
        """
        return False

    def get_or_create_batch(
        self,
        task_name: str,
        sampler_params: dict[str, Any],
        scan_ready_fn: Callable[[], list[int]],
        **kwargs: Any,
    ) -> tuple[list[int], Optional[list[int]]]:
        """Get or create a batch with state management for stateful samplers.
        
        This method is called for stateful samplers to handle coordinated data access.
        Stateless samplers can leave this unimplemented.
        
        Args:
            task_name: Name of the consumer task
            sampler_params: Parameters passed from the client
            scan_ready_fn: Function to scan for ready indices (production status already checked)
            **kwargs: Additional context (batch_size, data_fields, global_step, etc.)
        
        Returns:
            Tuple of (batch_indices, indices_to_mark_consumed)
            - batch_indices: List of indices for this request
            - indices_to_mark_consumed: List of indices to mark as consumed, or None if not ready yet
        
        Raises:
            NotImplementedError: If called on a stateless sampler
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is not stateful. "
            "Override is_stateful() to return True and implement get_or_create_batch()."
        )
    
    def mark_consumed(self, task_name: str, indices: list[int]) -> None:
        """Mark indices as consumed for a specific task.
        
        This is called by the sampler internally to track its own consumption state.
        
        Args:
            task_name: Name of the consumer task
            indices: List of indices to mark as consumed
        """
        if task_name not in self._consumption_state:
            self._consumption_state[task_name] = set()
        self._consumption_state[task_name].update(indices)
    
    def is_consumed(self, task_name: str, index: int) -> bool:
        """Check if an index has been consumed for a specific task.
        
        Args:
            task_name: Name of the consumer task
            index: Index to check
            
        Returns:
            True if the index has been consumed, False otherwise
        """
        return task_name in self._consumption_state and index in self._consumption_state[task_name]
    
    def get_consumed_indices(self, task_name: str) -> set[int]:
        """Get all consumed indices for a specific task.
        
        Args:
            task_name: Name of the consumer task
            
        Returns:
            Set of consumed indices
        """
        return self._consumption_state.get(task_name, set())
    
    def clear_consumed(self, task_name: str, indices: Optional[list[int]] = None) -> None:
        """Clear consumption state for specific indices or all indices of a task.
        
        Args:
            task_name: Name of the consumer task
            indices: Optional list of specific indices to clear. If None, clear all.
        """
        if task_name not in self._consumption_state:
            return
        
        if indices is None:
            self._consumption_state[task_name].clear()
        else:
            self._consumption_state[task_name].difference_update(indices)

    def __call__(self, *args: Any, **kwargs: Any) -> list[int]:
        """Allow samplers to be called directly."""
        return self.sample(*args, **kwargs)
