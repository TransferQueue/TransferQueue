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

from transfer_queue.sampler import BaseSampler


class SequentialSampler(BaseSampler):
    """Sequential sampler for basic data consumption patterns.

    This sampler implements sequential sampling without replacement, selecting samples
    from the beginning of the ready_indexes list in order. It's the default sampling
    strategy for TransferQueueController and provides simple, deterministic data consumption
    with minimal overhead.

    The sampler is ideal for standard supervised learning scenarios, data preprocessing
    pipelines, and any use case where ordered, predictable data consumption is preferred.
    It ensures each sample is consumed exactly once, maintaining a clean progression through
    the available data.

    This sampler is typically used as the default sampler in TransferQueueController:

    ```python
    from transfer_queue import (
        TransferQueueController,
        SequentialSampler,
        AsyncTransferQueueClient,
        TransferQueueStorageManagerFactory
    )

    # Default usage (SequentialSampler is the default)
    controller = TransferQueueController.remote()
    # or explicitly:
    controller = TransferQueueController.remote(sampler=SequentialSampler)
    controller_info = process_zmq_server_info(controller)

    client = AsyncTransferQueueClient(
        client_id="train_client",
        controller_info=controller_info,
    )

    # Initialize storage manager
    storage_config = {
        "controller_info": controller_info,
        "storage_unit_infos": {},
    }
    client.initialize_storage_manager("AsyncSimpleStorageManager", storage_config)

    # Get metadata - no sampling config needed
    meta = await client.async_get_meta(
        data_fields=["input_ids", "attention_mask", "labels"],
        batch_size=8,
        partition_id="train_0",
        task_name="supervised_training"
    )
    # Returns first 8 available samples in order
    ```
    """

    def __init__(
        self,
    ):
        """Initialize the SequentialSampler.

        SequentialSampler requires no initialization parameters and maintains
        minimal internal state for optimal performance.
        """
        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample indices sequentially from the beginning of ready_indexes.

        Selects the first `batch_size` elements from the ready_indexes list,
        maintaining the original order. If batch_size exceeds the available
        ready samples, all available samples are returned.

        Args:
            ready_indexes: List of global indices for which all required fields have been
                produced and samples are not labeled as consumed. The order in this list
                determines the sampling sequence.
            batch_size: Number of samples to select. If larger than available ready samples,
                all available samples will be returned.
            *args: Additional positional arguments (ignored in current implementation)
            **kwargs: Additional keyword arguments (ignored in current implementation)

        Returns:
            Tuple of (sampled_indexes, consumed_indexes):
            - sampled_indexes: List of selected global indices, length = min(batch_size, len(ready_indexes))
            - consumed_indexes: List of indices to mark as consumed, identical to sampled_indexes
              (without replacement semantics)

        Example:
            >>> sampler = SequentialSampler()
            >>> ready_indexes = [10, 20, 30, 40, 50]
            >>> sampled, consumed = sampler.sample(ready_indexes, 3)
            >>> sampled
            [10, 20, 30]
            >>> consumed
            [10, 20, 30]

            # Edge case: batch_size larger than available
            >>> sampled, consumed = sampler.sample([100, 200], 5)
            >>> sampled
            [100, 200]
            >>> consumed
            [100, 200]
        """
        sampled_indexes = ready_indexes[:batch_size]
        consumed_indexes = sampled_indexes

        return sampled_indexes, consumed_indexes
