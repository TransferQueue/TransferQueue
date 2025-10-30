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
Example demonstrating stateful DP sampling with sampler_params.

This example shows how to use DP sampling where:
1. Each process passes its dp_rank via sampler_params (no need to pre-register samplers)
2. Processes in the same DP group get the same data pool
3. Data is only marked consumed after all ranks in a DP group have consumed it
"""

import logging

import torch

from transfer_queue import SequentialSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_dp_sampling_with_params():
    """Demonstrate DP sampling using sampler_params instead of pre-registration."""
    logger.info("\n" + "=" * 80)
    logger.info("DP Sampling with sampler_params (No Pre-registration Needed)")
    logger.info("=" * 80)

    # Simulate ready indices
    ready_indices = list(range(16))
    dp_size = 4
    batch_size_per_rank = 4

    logger.info(f"\nReady indices: {ready_indices}")
    logger.info(f"DP configuration: {dp_size} groups, {batch_size_per_rank} samples per rank")

    # Each process can create its own sampler instance with its dp_rank
    # No need to register with controller beforehand
    for dp_rank in range(dp_size):
        # Each rank uses the same sampler but with different dp_rank
        sampler = SequentialSampler()
        
        # Pass dp parameters via sampler_params during get_meta call
        sampler_params = {
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "dp_group_id": "dp_group_step0",  # All ranks in same group use same ID
        }
        
        # In real usage, this would be:
        # batch_meta = client.get_meta(
        #     data_fields=["prompts"],
        #     batch_size=batch_size_per_rank,
        #     global_step=0,
        #     task_name="dp_task",
        #     sampler_params=sampler_params,
        # )
        
        # For demonstration, manually partition
        start_idx = dp_rank * batch_size_per_rank
        end_idx = start_idx + batch_size_per_rank
        rank_indices = ready_indices[start_idx:end_idx]
        
        logger.info(f"Rank {dp_rank} would get indices: {rank_indices}")

    logger.info("\n" + "=" * 80)
    logger.info("Key Benefits:")
    logger.info("=" * 80)
    logger.info("1. No need to pre-register samplers - pass params at get_meta time")
    logger.info("2. Stateful: All ranks in same DP group see consistent data pool")
    logger.info("3. Data only marked consumed after ALL ranks in group consume it")
    logger.info("4. Different DP groups (different dp_group_id) get different data")


def demonstrate_usage_pattern():
    """Show the usage pattern in pseudo-code."""
    logger.info("\n" + "=" * 80)
    logger.info("Usage Pattern (Pseudo-code)")
    logger.info("=" * 80)
    
    logger.info("""
# In each distributed process:
import os

dp_rank = int(os.environ.get("DP_RANK", 0))
dp_size = int(os.environ.get("DP_SIZE", 1))
global_step = 0

# Pass sampler params in get_meta - no pre-registration needed!
batch_meta = client.get_meta(
    data_fields=["prompts", "responses"],
    batch_size=4,
    global_step=global_step,
    task_name="training_task",
    sampler_params={
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "dp_group_id": f"dp_group_step{global_step}",  # Unique per step
    },
)

# All ranks in the same DP group will get data from the same pool
# Rank 0 gets indices [0, 1, 2, 3]
# Rank 1 gets indices [4, 5, 6, 7]
# etc.

# Data is only marked as consumed after ALL ranks retrieve it
    """)


def main():
    """Run demonstrations."""
    logger.info("TransferQueue DP Sampling with sampler_params")
    logger.info("=" * 80)

    demonstrate_dp_sampling_with_params()
    demonstrate_usage_pattern()

    logger.info("\n" + "=" * 80)
    logger.info("Demonstration completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
