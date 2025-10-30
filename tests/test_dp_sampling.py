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
import sys
from pathlib import Path

import pytest
import ray

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import TransferQueueController  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def ray_setup():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "1", "RAY_DEDUP_LOGS": "0"}},
        log_to_driver=True,
    )
    yield
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray has been shut down completely after test")


class TestDPSamplingStateful:
    """Test stateful DP sampling where ranks in the same DP group get the same data."""

    def test_dp_group_same_data(self, ray_setup):
        """Test that multiple ranks in the same DP group get the same data."""
        global_batch_size = 16
        num_n_samples = 1
        dp_size = 4
        batch_size_per_rank = 4

        controller = TransferQueueController.remote(
            global_batch_size=global_batch_size,
            num_global_batch=1,
            num_n_samples=num_n_samples,
        )

        # Simulate data production
        global_indexes = list(range(global_batch_size))
        data_fields = ["field_1"]
        ray.get(controller._update_production_status.remote(global_indexes, data_fields))

        # Simulate 4 ranks in the same DP group (same dp_group_id)
        # Each rank has dp_rank 0-3, but they're in the same DP group
        dp_group_id = "dp_group_step0"
        
        results = []
        for dp_rank in range(dp_size):
            metadata = ray.get(
                controller._get_metadata.remote(
                    data_fields=data_fields,
                    batch_size=batch_size_per_rank,
                    global_step=0,
                    mode="fetch",
                    task_name="dp_task",
                    sampler_params={
                        "dp_rank": dp_rank,
                        "dp_size": dp_size,
                        "dp_group_id": dp_group_id,
                    },
                )
            )
            results.append(metadata.global_indexes)
            logger.info(f"DP rank {dp_rank} got indices: {metadata.global_indexes}")

        # Verify that different ranks got different slices
        # but they're from the same total pool
        all_indices = []
        for indices in results:
            all_indices.extend(indices)
        
        # Each rank should have gotten unique data
        assert len(set(all_indices)) == len(all_indices), "Ranks should get non-overlapping data"
        
        # The data should be partitioned correctly
        assert results[0] == [0, 1, 2, 3]
        assert results[1] == [4, 5, 6, 7]
        assert results[2] == [8, 9, 10, 11]
        assert results[3] == [12, 13, 14, 15]

        ray.get(controller.clear.remote(0))

    def test_dp_different_groups_different_data(self, ray_setup):
        """Test that different DP groups get different data."""
        global_batch_size = 16
        num_n_samples = 1
        dp_size = 2
        batch_size_per_rank = 4

        controller = TransferQueueController.remote(
            global_batch_size=global_batch_size,
            num_global_batch=1,
            num_n_samples=num_n_samples,
        )

        # Simulate data production
        global_indexes = list(range(global_batch_size))
        data_fields = ["field_1"]
        ray.get(controller._update_production_status.remote(global_indexes, data_fields))

        # Two different DP groups
        group1_indices = []
        group2_indices = []

        # Group 1, Rank 0
        metadata = ray.get(
            controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size_per_rank,
                global_step=0,
                mode="fetch",
                task_name="dp_task",
                sampler_params={
                    "dp_rank": 0,
                    "dp_size": dp_size,
                    "dp_group_id": "dp_group1_step0",
                },
            )
        )
        group1_indices.extend(metadata.global_indexes)

        # Group 1, Rank 1
        metadata = ray.get(
            controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size_per_rank,
                global_step=0,
                mode="fetch",
                task_name="dp_task",
                sampler_params={
                    "dp_rank": 1,
                    "dp_size": dp_size,
                    "dp_group_id": "dp_group1_step0",  # Same group as above
                },
            )
        )
        group1_indices.extend(metadata.global_indexes)

        # Group 2, Rank 0
        metadata = ray.get(
            controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size_per_rank,
                global_step=0,
                mode="fetch",
                task_name="dp_task",
                sampler_params={
                    "dp_rank": 0,
                    "dp_size": dp_size,
                    "dp_group_id": "dp_group2_step0",  # Different group
                },
            )
        )
        group2_indices.extend(metadata.global_indexes)

        # Group 2, Rank 1
        metadata = ray.get(
            controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size_per_rank,
                global_step=0,
                mode="fetch",
                task_name="dp_task",
                sampler_params={
                    "dp_rank": 1,
                    "dp_size": dp_size,
                    "dp_group_id": "dp_group2_step0",  # Same as group 2
                },
            )
        )
        group2_indices.extend(metadata.global_indexes)

        logger.info(f"Group 1 indices: {group1_indices}")
        logger.info(f"Group 2 indices: {group2_indices}")

        # Different groups should get different data
        assert set(group1_indices) != set(group2_indices), "Different DP groups should get different data"
        
        # No overlap between groups
        assert len(set(group1_indices) & set(group2_indices)) == 0, "No overlap between DP groups"

        ray.get(controller.clear.remote(0))

    def test_dp_consumption_marked_after_all_ranks(self, ray_setup):
        """Test that data is only marked as consumed after all ranks in DP group consume it."""
        global_batch_size = 8
        num_n_samples = 1
        dp_size = 2
        batch_size_per_rank = 4

        controller = TransferQueueController.remote(
            global_batch_size=global_batch_size,
            num_global_batch=1,
            num_n_samples=num_n_samples,
        )

        # Simulate data production
        global_indexes = list(range(global_batch_size))
        data_fields = ["field_1"]
        ray.get(controller._update_production_status.remote(global_indexes, data_fields))

        dp_group_id = "dp_group_step0"
        task_name = "dp_task"

        # First rank consumes
        ray.get(
            controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size_per_rank,
                global_step=0,
                mode="fetch",
                task_name=task_name,
                sampler_params={
                    "dp_rank": 0,
                    "dp_size": dp_size,
                    "dp_group_id": dp_group_id,
                },
            )
        )

        # Check consumption status - should NOT be marked as consumed yet
        consumption_status = ray.get(controller.get_data_consumption_status.remote())
        consumed_count = consumption_status[task_name][:global_batch_size].sum().item()
        assert consumed_count == 0, "Data should not be marked as consumed until all ranks consume"

        # Second rank consumes
        ray.get(
            controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size_per_rank,
                global_step=0,
                mode="fetch",
                task_name=task_name,
                sampler_params={
                    "dp_rank": 1,
                    "dp_size": dp_size,
                    "dp_group_id": dp_group_id,
                },
            )
        )

        # Now all ranks have consumed, data should be marked as consumed
        consumption_status = ray.get(controller.get_data_consumption_status.remote())
        consumed_count = consumption_status[task_name][:global_batch_size].sum().item()
        assert consumed_count == global_batch_size, "Data should be marked as consumed after all ranks consume"

        ray.get(controller.clear.remote(0))
