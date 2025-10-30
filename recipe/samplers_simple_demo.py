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
Simple demonstration of custom samplers in TransferQueue.

This demonstrates the core sampler concepts without full integration.
"""

import torch

from transfer_queue import DPSampler, GRPOSampler, SequentialSampler

print("=" * 80)
print("TransferQueue Custom Samplers Demonstration")
print("=" * 80)

# Example 1: SequentialSampler (default behavior)
print("\n1. SequentialSampler (Default Behavior)")
print("-" * 80)
sampler = SequentialSampler()
ready_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
batch_size = 4

result = sampler.sample(ready_indices, batch_size)
print(f"Ready indices: {ready_indices}")
print(f"Batch size: {batch_size}")
print(f"Sampled indices: {result}")
print("Expected: [0, 1, 2, 3] (sequential order)")

# Example 2: GRPOSampler (group-based sampling)
print("\n2. GRPOSampler (Group-based Sampling for GRPO)")
print("-" * 80)
num_n_samples = 4  # 4 responses per prompt
sampler = GRPOSampler(num_n_samples=num_n_samples)

# Indices grouped by prompts: [0,1,2,3], [4,5,6,7], [8,9,10,11]
ready_indices = list(range(12))
batch_size = 8  # 2 complete groups

result = sampler.sample(ready_indices, batch_size)
print(f"Ready indices (3 groups of 4): {ready_indices}")
print(f"Batch size: {batch_size} (2 groups × 4 samples)")
print(f"Sampled indices: {result}")
print("Expected: [0,1,2,3, 4,5,6,7] (2 complete groups)")

# Test filtering with GRPO sampler
print("\nGRPO filtering with partial data:")
production_status = torch.zeros(12, 2, dtype=torch.int8)
# Group 0 [0,1,2,3]: all ready
production_status[0:4, 0] = 1
production_status[0:4, 1] = 1
# Group 1 [4,5,6,7]: only partially ready (missing index 6)
production_status[4:6, 0] = 1
production_status[4:6, 1] = 1
production_status[7, 0] = 1
production_status[7, 1] = 1
# Group 2 [8,9,10,11]: all ready
production_status[8:12, 0] = 1
production_status[8:12, 1] = 1

consumption_status = torch.zeros(12, dtype=torch.int8)
all_indices = list(range(12))
data_fields = ["field_0", "field_1"]
field_mapping = {"field_0": 0, "field_1": 1}

ready_indices = sampler.filter_ready_indices(
    all_indices, production_status, consumption_status, data_fields, field_mapping
)
print(f"Filtered ready indices: {ready_indices}")
print("Expected: [0,1,2,3, 8,9,10,11] (only groups 0 and 2 are complete)")

# Example 3: DPSampler (data parallel aware sampling)
print("\n3. DPSampler (Data Parallel Aware Sampling)")
print("-" * 80)
dp_size = 4  # 4 DP groups
batch_size_per_dp = 3  # Each DP group gets 3 samples
ready_indices = list(range(12))  # 12 total samples

print(f"Total ready indices: {ready_indices}")
print(f"DP configuration: {dp_size} groups, {batch_size_per_dp} samples each")
print("\nEach DP rank gets different, non-overlapping data:")

for dp_rank in range(dp_size):
    sampler = DPSampler(dp_rank=dp_rank, dp_size=dp_size)
    result = sampler.sample(ready_indices, batch_size_per_dp)
    print(f"  DP rank {dp_rank}: {result}")

print("\nExpected partitioning:")
print("  DP rank 0: [0, 1, 2]")
print("  DP rank 1: [3, 4, 5]")
print("  DP rank 2: [6, 7, 8]")
print("  DP rank 3: [9, 10, 11]")

# Example 4: Using samplers with TransferQueueController
print("\n4. Integration with TransferQueueController")
print("-" * 80)
print("""
To use samplers with TransferQueueController:

# Option 1: Set default sampler at initialization
controller = TransferQueueController.remote(
    global_batch_size=8,
    num_global_batch=1,
    num_n_samples=4,
    sampler=GRPOSampler(num_n_samples=4),  # All tasks use GRPO by default
)

# Option 2: Register task-specific samplers
controller = TransferQueueController.remote(
    global_batch_size=8,
    num_global_batch=1,
    num_n_samples=4,
)

# Register GRPO sampler for GRPO task
ray.get(controller.register_sampler.remote(
    task_name="grpo_task",
    sampler=GRPOSampler(num_n_samples=4),
))

# Register DP sampler for DP task
ray.get(controller.register_sampler.remote(
    task_name="dp_task",
    sampler=DPSampler(dp_rank=0, dp_size=2),
))

# Fetch data with specific samplers
batch_meta = client.get_meta(
    data_fields=["prompts"],
    batch_size=8,
    global_step=0,
    task_name="grpo_task",  # Uses GRPOSampler
    get_n_samples=True,
)
""")

print("\n" + "=" * 80)
print("Demonstration completed successfully!")
print("=" * 80)
