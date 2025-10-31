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
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transfer_queue import TransferQueueController  # noqa: E402
from transfer_queue.controller import TQ_INIT_FIELD_NUM  # noqa: E402
from transfer_queue.utils.utils import ProductionStatus  # noqa: E402


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


class TestTransferQueueController:
    def test_controller_with_single_partition(self, ray_setup):
        gbs = 8
        num_n_samples = 4

        tq_controller = TransferQueueController.remote()

        # Test get metadata in insert mode
        partition_id = "train_0"
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="insert",
                get_n_samples=True,
            )
        )

        assert metadata.global_indexes == list(range(gbs * num_n_samples))
        assert metadata.samples[0].partition_id == "train_0"
        assert sum([int(sample.fields.get("prompt_ids").production_status) for sample in metadata.samples]) == int(
            ProductionStatus.NOT_PRODUCED
        )
        assert sum([int(sample.fields.get("attention_mask").production_status) for sample in metadata.samples]) == int(
            ProductionStatus.NOT_PRODUCED
        )
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id))
        assert partition_index_range == set(range(gbs * num_n_samples))

        print("✓ Initial get metadata correct")

        # Test update production status
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                sample_indices=metadata.global_indexes,
                field_names=metadata.field_names,
            )
        )
        assert success
        partition = ray.get(tq_controller.get_partition.remote(partition_id))
        assert partition.production_status is not None
        assert partition.production_status.size(0) == gbs * num_n_samples
        assert partition.production_status.size(1) == TQ_INIT_FIELD_NUM
        assert torch.equal(
            sum(partition.production_status[:, :len(data_fields)]),
            torch.Tensor([gbs*num_n_samples, gbs*num_n_samples]),
        )
        assert torch.equal(
            sum(partition.production_status[:, len(data_fields):]),
            torch.zeros(1 * (TQ_INIT_FIELD_NUM - len(data_fields))),
        )

        print(f"✓ Updated production status for partition {partition_id}")

        # Test get metadate in fetch mode
        gen_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=["prompt_ids"],
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="fetch",
                task_name="generate_sequences",
                get_n_samples=False,
            )
        )
        assert gen_meta.global_indexes == list(range(gbs * num_n_samples))
        assert gen_meta.samples[0].partition_id == "train_0"
        assert gen_meta.field_names == ["prompt_ids"]
        partition = ray.get(tq_controller.get_partition.remote(partition_id))
        assert torch.equal(partition.consumption_status["generate_sequences"], torch.ones(gbs * num_n_samples))
        print("✓ Get metadata in fetch mode correct")

        # Test get clear meta
        clear_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=[],
                partition_id=partition_id,
                mode="insert",
            )
        )
        assert clear_meta.global_indexes == list(range(gbs * num_n_samples))
        assert [sample.fields for sample in clear_meta.samples] == [{}] * (gbs * num_n_samples)
        print("✓ Clear metadata correct")

        # Test clear
        ray.get(tq_controller.clear.remote(partition_id))
        partition = ray.get(tq_controller.get_partition.remote(partition_id))
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id))
        assert partition_index_range == set()
        assert torch.all(partition.production_status == 0)
        assert torch.all(partition.consumption_status["generate_sequences"] == 0)
        print("✓ Clear correct")
