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

from transfer_queue import TransferQueueController  # noqa: E402
from transfer_queue.controller import TQ_INIT_FIELD_NUM  # noqa: E402

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


@pytest.fixture(scope="function")
def setup_teardown_transfer_queue_controller(ray_setup):
    # Used as the offset for the global index to distinguish which global step the data corresponds to
    global_batch_size = 8
    num_global_batch = 2
    num_n_samples = 2

    tq_controller = TransferQueueController.remote(
        global_batch_size=global_batch_size,
        num_global_batch=num_global_batch,
        num_n_samples=num_n_samples,
    )
    yield tq_controller, global_batch_size, num_global_batch, num_n_samples
    ray.get(tq_controller.clear.remote(0))


class TestTransferQueueController:
    def test_update_production_status(self, setup_teardown_transfer_queue_controller):
        tq_controller, global_batch_size, num_global_batch, num_n_samples = setup_teardown_transfer_queue_controller

        total_storage_size = global_batch_size * num_global_batch * num_n_samples
        # Initialize get_data_production_status and filed_name_mapping
        init_update_production_status = torch.zeros(total_storage_size, TQ_INIT_FIELD_NUM, dtype=torch.int8)
        assert torch.equal(ray.get(tq_controller.get_data_production_status.remote()), init_update_production_status)
        assert ray.get(tq_controller.get_field_name_mapping.remote()) == {}

        columns_list = ["test_prompts"]
        global_indexes = list(range(global_batch_size * num_n_samples))

        # update production status
        tq_controller._update_production_status.remote(global_indexes, columns_list)
        new_field_name_mapping = ray.get(tq_controller.get_field_name_mapping.remote())
        assert new_field_name_mapping["test_prompts"] == 0

        new_data_production_status = ray.get(tq_controller.get_data_production_status.remote())
        assert new_data_production_status[:, 0][: len(global_indexes)].sum() == len(global_indexes)

    def test_data_consumption_status(self, setup_teardown_transfer_queue_controller):
        tq_controller, global_batch_size, num_global_batch, num_n_samples = setup_teardown_transfer_queue_controller
        total_storage_size = global_batch_size * num_global_batch * num_n_samples

        init_data_consumption_status = {}
        assert ray.get(tq_controller.get_data_consumption_status.remote()) == init_data_consumption_status

        task_name = "test_task1"
        ray.get(tq_controller._get_consumption_status.remote(task_name))
        new_data_consumption_status = ray.get(tq_controller.get_data_consumption_status.remote())
        assert torch.equal(new_data_consumption_status[task_name], torch.zeros(total_storage_size, dtype=torch.int8))

    def test_get_prompt_metadata(self, setup_teardown_transfer_queue_controller):
        tq_controller, global_batch_size, _, n_samples = setup_teardown_transfer_queue_controller

        data_fields = ["test_prompts"]
        global_step = 5

        metadata = ray.get(
            tq_controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=global_batch_size * n_samples,
                global_step=global_step,
                mode="insert",
            )
        )
        metadata.reorder([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        assert metadata.global_indexes == [
            31,
            30,
            29,
            28,
            27,
            26,
            25,
            24,
            23,
            22,
            21,
            20,
            19,
            18,
            17,
            16,
        ]

    # TODO: Test case where multiple clients concurrently read datameta from a single controller,
    #  and each client receives the correct response
