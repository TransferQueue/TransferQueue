#!/usr/bin/env python3
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

"""
Example demonstrating the use of custom samplers in TransferQueue.

This example shows three sampling strategies:
1. SequentialSampler (default): Sequential sampling maintaining backward compatibility
2. GRPOSampler: Group-based sampling for GRPO (Group Relative Policy Optimization)
3. DPSampler: Data Parallel aware sampling for distributed training

Run this example:
    python recipe/samplers_example.py
"""

import logging

import ray
import torch
from tensordict import TensorDict

from transfer_queue import (
    DPSampler,
    GRPOSampler,
    SequentialSampler,
    TransferQueueClient,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.storage import AsyncSimpleStorageManager, SimpleStorageUnit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_sequential_sampler():
    """Example 1: Using SequentialSampler (default behavior)."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 1: SequentialSampler (Default)")
    logger.info("=" * 80)

    ray.init(ignore_reinit_error=True)

    # Create controller with default sampler (SequentialSampler)
    global_batch_size = 8
    num_n_samples = 2
    controller = TransferQueueController.remote(
        global_batch_size=global_batch_size,
        num_global_batch=1,
        num_n_samples=num_n_samples,
    )

    # Create storage
    storage = SimpleStorageUnit.remote(
        storage_unit_size=global_batch_size * num_n_samples,
    )

    # Create client
    client = TransferQueueClient(
        client_id="sequential_client",
        controller_info=process_zmq_server_info(controller),
    )

    client.initialize_storage_manager(
        manager_type="AsyncSimpleStorageManager",
        config={
            "controller_info": process_zmq_server_info(controller),
            "storage_unit_infos": process_zmq_server_info({"storage_0": storage}),
        },
    )

    # Write data
    prompts = torch.randn(global_batch_size * num_n_samples, 10)
    data = TensorDict({"prompts": prompts}, batch_size=[global_batch_size * num_n_samples])
    client.put(data=data, global_step=0)

    # Read data using sequential sampler
    batch_meta = client.get_meta(
        data_fields=["prompts"],
        batch_size=4,
        global_step=0,
        task_name="sequential_task",
    )

    logger.info(f"Sequential sampler returned indices: {batch_meta.global_indexes}")
    logger.info("Expected: [0, 1, 2, 3] (sequential order)")

    ray.shutdown()


def example_grpo_sampler():
    """Example 2: Using GRPOSampler for grouped sampling."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: GRPOSampler (Group-based sampling for GRPO)")
    logger.info("=" * 80)

    ray.init(ignore_reinit_error=True)

    # Create controller with GRPO sampler
    global_batch_size = 8
    num_n_samples = 4  # 4 responses per prompt
    controller = TransferQueueController.remote(
        global_batch_size=global_batch_size,
        num_global_batch=1,
        num_n_samples=num_n_samples,
        sampler=GRPOSampler(num_n_samples=num_n_samples),
    )

    # Create storage
    storage = SimpleStorageUnit.remote(
        storage_unit_size=global_batch_size * num_n_samples,
    )

    # Create client
    client = TransferQueueClient(
        client_id="grpo_client",
        controller_info=process_zmq_server_info(controller),
    )

    client.initialize_storage_manager(
        manager_type="AsyncSimpleStorageManager",
        config={
            "controller_info": process_zmq_server_info(controller),
            "storage_unit_infos": process_zmq_server_info({"storage_0": storage}),
        },
    )

    # Write data - prompts repeated for n_samples
    original_prompts = torch.randn(global_batch_size, 10)
    prompts_repeated = torch.repeat_interleave(original_prompts, num_n_samples, dim=0)
    data = TensorDict({"prompts": prompts_repeated}, batch_size=[global_batch_size * num_n_samples])
    client.put(data=data, global_step=0)

    # Read data using GRPO sampler - must request multiples of num_n_samples
    batch_size = 8  # 2 groups of 4
    batch_meta = client.get_meta(
        data_fields=["prompts"],
        batch_size=batch_size,
        global_step=0,
        task_name="grpo_task",
        get_n_samples=True,
    )

    logger.info(f"GRPO sampler returned {len(batch_meta.global_indexes)} indices in {batch_size//num_n_samples} groups")
    logger.info(f"Indices: {batch_meta.global_indexes}")
    logger.info(f"Expected: Complete groups like [0,1,2,3, 4,5,6,7]")

    ray.shutdown()


def example_dp_sampler():
    """Example 3: Using DPSampler for data parallel training."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: DPSampler (Data Parallel aware sampling)")
    logger.info("=" * 80)

    ray.init(ignore_reinit_error=True)

    # Create controller (initially with default sampler, will register DP samplers per task)
    global_batch_size = 16
    num_n_samples = 1
    controller = TransferQueueController.remote(
        global_batch_size=global_batch_size,
        num_global_batch=1,
        num_n_samples=num_n_samples,
    )

    # Simulate 4 DP groups
    dp_size = 4
    batch_size_per_dp = 4  # Each DP group gets 4 samples

    # Register DP samplers for different ranks
    for dp_rank in range(dp_size):
        task_name = f"dp_task_rank_{dp_rank}"
        ray.get(
            controller.register_sampler.remote(
                task_name=task_name,
                sampler=DPSampler(dp_rank=dp_rank, dp_size=dp_size),
            )
        )

    # Create storage
    storage = SimpleStorageUnit.remote(
        storage_unit_size=global_batch_size * num_n_samples,
    )

    # Create client
    client = TransferQueueClient(
        client_id="dp_client",
        controller_info=process_zmq_server_info(controller),
    )

    client.initialize_storage_manager(
        manager_type="AsyncSimpleStorageManager",
        config={
            "controller_info": process_zmq_server_info(controller),
            "storage_unit_infos": process_zmq_server_info({"storage_0": storage}),
        },
    )

    # Write data
    prompts = torch.randn(global_batch_size, 10)
    data = TensorDict({"prompts": prompts}, batch_size=[global_batch_size])
    client.put(data=data, global_step=0)

    # Simulate different DP ranks reading data
    logger.info("\nSimulating 4 DP ranks fetching their respective data:")
    for dp_rank in range(dp_size):
        task_name = f"dp_task_rank_{dp_rank}"
        batch_meta = client.get_meta(
            data_fields=["prompts"],
            batch_size=batch_size_per_dp,
            global_step=0,
            task_name=task_name,
        )

        logger.info(f"DP rank {dp_rank} got indices: {batch_meta.global_indexes}")

    logger.info("\nNote: Each DP rank gets different, non-overlapping data")

    ray.shutdown()


def example_mixed_samplers():
    """Example 4: Using different samplers for different tasks."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Mixed Samplers (Different samplers for different tasks)")
    logger.info("=" * 80)

    ray.init(ignore_reinit_error=True)

    # Create controller with default sampler
    global_batch_size = 16
    num_n_samples = 4
    controller = TransferQueueController.remote(
        global_batch_size=global_batch_size,
        num_global_batch=1,
        num_n_samples=num_n_samples,
    )

    # Register GRPO sampler for GRPO task
    ray.get(
        controller.register_sampler.remote(
            task_name="grpo_task",
            sampler=GRPOSampler(num_n_samples=num_n_samples),
        )
    )

    # Register DP sampler for DP task (rank 0 of 2 DP groups)
    ray.get(
        controller.register_sampler.remote(
            task_name="dp_task",
            sampler=DPSampler(dp_rank=0, dp_size=2),
        )
    )

    # Create storage
    storage = SimpleStorageUnit.remote(
        storage_unit_size=global_batch_size * num_n_samples,
    )

    # Create client
    client = TransferQueueClient(
        client_id="mixed_client",
        controller_info=process_zmq_server_info(controller),
    )

    client.initialize_storage_manager(
        manager_type="AsyncSimpleStorageManager",
        config={
            "controller_info": process_zmq_server_info(controller),
            "storage_unit_infos": process_zmq_server_info({"storage_0": storage}),
        },
    )

    # Write data
    original_prompts = torch.randn(global_batch_size, 10)
    prompts_repeated = torch.repeat_interleave(original_prompts, num_n_samples, dim=0)
    data = TensorDict({"prompts": prompts_repeated}, batch_size=[global_batch_size * num_n_samples])
    client.put(data=data, global_step=0)

    # Task 1: GRPO task with group-based sampling
    batch_meta_grpo = client.get_meta(
        data_fields=["prompts"],
        batch_size=8,  # 2 groups of 4
        global_step=0,
        task_name="grpo_task",
        get_n_samples=True,
    )
    logger.info(f"GRPO task got indices: {batch_meta_grpo.global_indexes}")

    # Task 2: DP task with DP-aware sampling (DP rank 0 gets first half)
    batch_meta_dp = client.get_meta(
        data_fields=["prompts"],
        batch_size=4,
        global_step=0,
        task_name="dp_task",
    )
    logger.info(f"DP task (rank 0) got indices: {batch_meta_dp.global_indexes}")

    # Task 3: Default task with sequential sampling
    batch_meta_default = client.get_meta(
        data_fields=["prompts"],
        batch_size=4,
        global_step=0,
        task_name="default_task",
    )
    logger.info(f"Default task got indices: {batch_meta_default.global_indexes}")

    ray.shutdown()


def main():
    """Run all examples."""
    logger.info("TransferQueue Custom Samplers Examples")
    logger.info("=" * 80)

    try:
        example_sequential_sampler()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")

    try:
        example_grpo_sampler()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")

    try:
        example_dp_sampler()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")

    try:
        example_mixed_samplers()
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("All examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
