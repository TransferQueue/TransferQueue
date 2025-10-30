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
import torch

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    DPSampler,
    GRPOSampler,
    SequentialSampler,
    TransferQueueController,
)

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


class TestSequentialSampler:
    """Test SequentialSampler maintains backward compatibility."""

    def test_sample_basic(self):
        """Test basic sequential sampling."""
        sampler = SequentialSampler()
        ready_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 4

        result = sampler.sample(ready_indices, batch_size)
        assert result == [0, 1, 2, 3]

    def test_sample_insufficient_data(self):
        """Test error handling when insufficient data available."""
        sampler = SequentialSampler()
        ready_indices = [0, 1, 2]
        batch_size = 4

        with pytest.raises(ValueError, match="Insufficient ready indices"):
            sampler.sample(ready_indices, batch_size)

    def test_filter_ready_indices(self):
        """Test filtering logic for ready indices."""
        sampler = SequentialSampler()

        # Create mock production and consumption status
        production_status = torch.zeros(10, 3, dtype=torch.int8)
        production_status[0:4, 0] = 1  # Field 0 ready for indices 0-3
        production_status[0:4, 1] = 1  # Field 1 ready for indices 0-3
        production_status[2:6, 0] = 1  # Field 0 ready for indices 2-5
        production_status[2:6, 1] = 1  # Field 1 ready for indices 2-5

        consumption_status = torch.zeros(10, dtype=torch.int8)

        all_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        data_fields = ["field_0", "field_1"]
        field_mapping = {"field_0": 0, "field_1": 1}

        ready_indices = sampler.filter_ready_indices(
            all_indices, production_status, consumption_status, data_fields, field_mapping
        )

        # Indices 0-5 have both fields ready (union of ranges [0,4) and [2,6))
        assert ready_indices == [0, 1, 2, 3, 4, 5]


class TestGRPOSampler:
    """Test GRPOSampler for group-based sampling."""

    def test_sample_basic(self):
        """Test basic GRPO sampling with groups."""
        num_n_samples = 4
        sampler = GRPOSampler(num_n_samples=num_n_samples)

        # Ready indices are grouped: [0,1,2,3], [4,5,6,7], [8,9,10,11]
        ready_indices = list(range(12))
        batch_size = 8  # 2 groups

        result = sampler.sample(ready_indices, batch_size)
        assert result == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_sample_invalid_batch_size(self):
        """Test error when batch_size not divisible by num_n_samples."""
        sampler = GRPOSampler(num_n_samples=4)
        ready_indices = list(range(12))
        batch_size = 5  # Not divisible by 4

        with pytest.raises(ValueError, match="must be divisible by"):
            sampler.sample(ready_indices, batch_size)

    def test_filter_ready_indices_complete_groups(self):
        """Test filtering returns only complete groups."""
        num_n_samples = 3
        sampler = GRPOSampler(num_n_samples=num_n_samples)

        # Create production status where only some groups are complete
        production_status = torch.zeros(12, 2, dtype=torch.int8)
        # Group 0 [0,1,2]: all ready
        production_status[0:3, 0] = 1
        production_status[0:3, 1] = 1
        # Group 1 [3,4,5]: only partially ready
        production_status[3:5, 0] = 1
        production_status[3:5, 1] = 1
        # Group 2 [6,7,8]: all ready
        production_status[6:9, 0] = 1
        production_status[6:9, 1] = 1
        # Group 3 [9,10,11]: not ready

        consumption_status = torch.zeros(12, dtype=torch.int8)

        all_indices = list(range(12))
        data_fields = ["field_0", "field_1"]
        field_mapping = {"field_0": 0, "field_1": 1}

        ready_indices = sampler.filter_ready_indices(
            all_indices, production_status, consumption_status, data_fields, field_mapping
        )

        # Only groups 0 and 2 are complete
        assert ready_indices == [0, 1, 2, 6, 7, 8]


class TestDPSampler:
    """Test DPSampler for data parallel aware sampling."""

    def test_sample_basic(self):
        """Test basic DP sampling with multiple DP groups."""
        # 4 DP groups, this process is rank 1
        sampler = DPSampler(dp_rank=1, dp_size=4)

        ready_indices = list(range(16))  # 16 samples total
        batch_size = 4  # Each DP group gets 4 samples

        result = sampler.sample(ready_indices, batch_size)
        # DP rank 1 should get indices 4-7
        assert result == [4, 5, 6, 7]

    def test_sample_different_ranks(self):
        """Test that different DP ranks get different data."""
        ready_indices = list(range(12))
        batch_size = 3

        sampler_rank0 = DPSampler(dp_rank=0, dp_size=4)
        sampler_rank1 = DPSampler(dp_rank=1, dp_size=4)
        sampler_rank2 = DPSampler(dp_rank=2, dp_size=4)

        result0 = sampler_rank0.sample(ready_indices, batch_size)
        result1 = sampler_rank1.sample(ready_indices, batch_size)
        result2 = sampler_rank2.sample(ready_indices, batch_size)

        assert result0 == [0, 1, 2]
        assert result1 == [3, 4, 5]
        assert result2 == [6, 7, 8]

        # Ensure no overlap
        assert set(result0) & set(result1) == set()
        assert set(result1) & set(result2) == set()

    def test_sample_insufficient_data(self):
        """Test error when insufficient data for all DP groups."""
        sampler = DPSampler(dp_rank=0, dp_size=4)
        ready_indices = [0, 1, 2]  # Only 3 samples, need 4*batch_size
        batch_size = 2

        with pytest.raises(ValueError, match="Insufficient ready indices"):
            sampler.sample(ready_indices, batch_size)

    def test_invalid_dp_rank(self):
        """Test error handling for invalid DP configuration."""
        with pytest.raises(ValueError):
            DPSampler(dp_rank=4, dp_size=4)  # dp_rank should be < dp_size

        with pytest.raises(ValueError):
            DPSampler(dp_rank=-1, dp_size=4)  # negative dp_rank


class TestControllerWithSamplers:
    """Test TransferQueueController with different samplers."""

    @pytest.fixture(scope="function")
    def setup_controller_with_sequential_sampler(self, ray_setup):
        """Setup controller with sequential sampler (default)."""
        global_batch_size = 8
        num_global_batch = 2
        num_n_samples = 2

        # Default sampler (SequentialSampler)
        tq_controller = TransferQueueController.remote(
            global_batch_size=global_batch_size,
            num_global_batch=num_global_batch,
            num_n_samples=num_n_samples,
        )
        yield tq_controller, global_batch_size, num_global_batch, num_n_samples
        ray.get(tq_controller.clear.remote(0))

    @pytest.fixture(scope="function")
    def setup_controller_with_grpo_sampler(self, ray_setup):
        """Setup controller with GRPO sampler."""
        global_batch_size = 8
        num_global_batch = 2
        num_n_samples = 4

        # GRPO sampler
        tq_controller = TransferQueueController.remote(
            global_batch_size=global_batch_size,
            num_global_batch=num_global_batch,
            num_n_samples=num_n_samples,
            sampler=GRPOSampler(num_n_samples=num_n_samples),
        )
        yield tq_controller, global_batch_size, num_global_batch, num_n_samples
        ray.get(tq_controller.clear.remote(0))

    def test_controller_default_sampler(self, setup_controller_with_sequential_sampler):
        """Test controller works with default sequential sampler."""
        tq_controller, global_batch_size, _, num_n_samples = setup_controller_with_sequential_sampler

        # Simulate data production
        global_indexes = list(range(global_batch_size * num_n_samples))
        data_fields = ["field_1"]

        ray.get(tq_controller._update_production_status.remote(global_indexes, data_fields))

        # Fetch metadata using default sampler
        metadata = ray.get(
            tq_controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=4,
                global_step=0,
                mode="fetch",
                task_name="test_task",
                get_n_samples=False,
            )
        )

        assert len(metadata.samples) == 4
        # Sequential sampler should return first 4 samples
        assert metadata.global_indexes == [0, 1, 2, 3]

    def test_controller_grpo_sampler(self, setup_controller_with_grpo_sampler):
        """Test controller works with GRPO sampler."""
        tq_controller, global_batch_size, _, num_n_samples = setup_controller_with_grpo_sampler

        # Simulate data production for all samples
        global_indexes = list(range(global_batch_size * num_n_samples))
        data_fields = ["field_1", "field_2"]

        ray.get(tq_controller._update_production_status.remote(global_indexes, data_fields))

        # Fetch metadata using GRPO sampler - must request multiple of num_n_samples
        batch_size = 8  # 2 groups of 4
        metadata = ray.get(
            tq_controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size,
                global_step=0,
                mode="fetch",
                task_name="grpo_task",
                get_n_samples=True,
            )
        )

        assert len(metadata.samples) == 8
        # Should get 2 complete groups
        assert metadata.global_indexes == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_controller_register_sampler(self, ray_setup):
        """Test registering task-specific samplers."""
        global_batch_size = 8
        num_global_batch = 2
        num_n_samples = 4

        # Create controller with default sampler
        tq_controller = TransferQueueController.remote(
            global_batch_size=global_batch_size,
            num_global_batch=num_global_batch,
            num_n_samples=num_n_samples,
        )

        # Register GRPO sampler for specific task
        ray.get(
            tq_controller.register_sampler.remote(
                task_name="grpo_task",
                sampler=GRPOSampler(num_n_samples=num_n_samples),
            )
        )

        # Simulate data production
        global_indexes = list(range(global_batch_size * num_n_samples))
        data_fields = ["field_1"]
        ray.get(tq_controller._update_production_status.remote(global_indexes, data_fields))

        # Fetch with GRPO task - should use GRPO sampler
        metadata_grpo = ray.get(
            tq_controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=8,
                global_step=0,
                mode="fetch",
                task_name="grpo_task",
                get_n_samples=True,
            )
        )

        assert len(metadata_grpo.samples) == 8

        # Fetch with different task - should use default sequential sampler
        metadata_default = ray.get(
            tq_controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=4,
                global_step=0,
                mode="fetch",
                task_name="other_task",
                get_n_samples=False,
            )
        )

        assert len(metadata_default.samples) == 4

        ray.get(tq_controller.clear.remote(0))
        ray.shutdown()
