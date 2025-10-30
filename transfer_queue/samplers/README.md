# TransferQueue Samplers

Custom samplers allow you to control how data is consumed from the TransferQueue. This decouples the controller from specific consumption patterns, enabling flexible support for various algorithms and distributed training strategies.

## Overview

The sampler interface is inspired by [torchrl.data.ReplayBuffer](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.ReplayBuffer.html), providing a familiar abstraction for controlling data sampling behavior.

## Available Samplers

### 1. SequentialSampler (Default)

Maintains backward compatibility by sampling data in sequential order.

```python
from transfer_queue import TransferQueueController, SequentialSampler

controller = TransferQueueController.remote(
    global_batch_size=8,
    num_global_batch=1,
    num_n_samples=1,
    sampler=SequentialSampler(),  # Optional, this is the default
)
```

**Use case:** Standard sequential data consumption.

### 2. GRPOSampler

Groups samples by prompts and ensures all `n_samples` for each prompt are ready before consumption. This is essential for algorithms like GRPO (Group Relative Policy Optimization) where multiple responses per prompt need to be processed together.

```python
from transfer_queue import TransferQueueController, GRPOSampler

num_n_samples = 4  # 4 responses per prompt
controller = TransferQueueController.remote(
    global_batch_size=8,
    num_global_batch=1,
    num_n_samples=num_n_samples,
    sampler=GRPOSampler(num_n_samples=num_n_samples),
)
```

**Use case:** GRPO and other algorithms that require synchronized group sampling.

**Key features:**
- Ensures all `n_samples` in a group are ready before any can be consumed
- Returns complete groups only
- `batch_size` must be divisible by `num_n_samples`

### 3. DPSampler

Partitions data across Data Parallel (DP) groups, ensuring ranks within the same DP group get data from the same pool while different DP groups get different data.

**Recommended Approach: Pass sampler_params in get_meta**

Instead of pre-registering samplers, pass DP parameters directly in the `get_meta` call:

```python
from transfer_queue import TransferQueueClient

# In each distributed process
client = TransferQueueClient(...)

# Pass DP parameters via sampler_params
batch_meta = client.get_meta(
    data_fields=["prompts"],
    batch_size=4,
    global_step=0,
    task_name="training_task",
    sampler_params={
        "dp_rank": 0,           # This rank's position in DP domain
        "dp_size": 4,           # Total number of DP groups
        "dp_group_id": "group_step0",  # Unique ID for this DP group
    },
)
```

**Alternative: Pre-register DPSampler (less flexible)**

```python
from transfer_queue import TransferQueueController, DPSampler

controller = TransferQueueController.remote(
    global_batch_size=16,
    num_global_batch=1,
    num_n_samples=1,
)

# Register DP sampler for this rank
controller.register_sampler.remote(
    task_name="training_task",
    sampler=DPSampler(dp_rank=0, dp_size=4),
)
```

**Use case:** Distributed training with data parallelism.

**Key features:**
- **Stateful:** Ranks in the same DP group (same `dp_group_id`) get data from the same pool
- **Consistent:** Data is only marked consumed after ALL ranks in a DP group have consumed it
- **Partitioned:** Different DP groups get non-overlapping data
- **Flexible:** No need to pre-register samplers when using `sampler_params`

## Usage Patterns

### Pattern 1: Pass Sampler Parameters in get_meta (Recommended for DP)

For DP sampling, the recommended approach is to pass sampler parameters directly in the `get_meta` call. This is more flexible as each process can specify its own parameters without pre-registration:

```python
from transfer_queue import TransferQueueClient
import os

# Get DP configuration from environment
dp_rank = int(os.environ.get("DP_RANK", 0))
dp_size = int(os.environ.get("DP_SIZE", 1))

client = TransferQueueClient(...)

# Each process passes its own DP parameters
batch_meta = client.get_meta(
    data_fields=["prompts", "responses"],
    batch_size=4,
    global_step=0,
    task_name="training_task",
    sampler_params={
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "dp_group_id": f"dp_group_step{global_step}",  # Must be same for all ranks in group
    },
)
```

**Benefits:**
- No need to pre-register samplers on the controller
- Each process independently specifies its parameters
- Cleaner separation between controller and client logic
- Ranks in the same DP group (same `dp_group_id`) automatically get consistent data

### Pattern 2: Default Sampler at Initialization

Set a default sampler for all tasks:

```python
from transfer_queue import TransferQueueController, GRPOSampler

controller = TransferQueueController.remote(
    global_batch_size=8,
    num_global_batch=1,
    num_n_samples=4,
    sampler=GRPOSampler(num_n_samples=4),  # All tasks use GRPO
)
```

### Pattern 3: Task-Specific Samplers via Pre-registration

Register different samplers for different tasks:

```python
from transfer_queue import TransferQueueController, GRPOSampler, DPSampler

# Create controller with default sampler
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

# Tasks use their registered samplers
# "grpo_task" uses GRPOSampler
# "dp_task" uses DPSampler
# Other tasks use default SequentialSampler
```

### Pattern 4: Callable Sampler Constructors

Pass sampler constructors for lazy initialization:

```python
controller = TransferQueueController.remote(
    global_batch_size=8,
    num_global_batch=1,
    num_n_samples=4,
    sampler=lambda: GRPOSampler(num_n_samples=4),  # Callable
)
```

## Client Usage

When fetching data, specify the task name to use the appropriate sampler:

```python
from transfer_queue import TransferQueueClient

client = TransferQueueClient(
    client_id="my_client",
    controller_info=controller_info,
)

# Fetch data using the sampler registered for "grpo_task"
batch_meta = client.get_meta(
    data_fields=["prompts", "responses"],
    batch_size=8,
    global_step=0,
    task_name="grpo_task",  # Uses GRPOSampler
    get_n_samples=True,
)
```

## Creating Custom Samplers

To create your own sampler, inherit from `BaseSampler` and implement two methods:

```python
from transfer_queue.samplers import BaseSampler
import torch

class MyCustomSampler(BaseSampler):
    def sample(self, ready_indices, batch_size, **kwargs):
        """Select batch_size indices from ready_indices."""
        # Your sampling logic here
        return ready_indices[:batch_size]
    
    def filter_ready_indices(
        self, 
        all_indices,
        production_status,
        consumption_status,
        data_fields,
        field_mapping,
        **kwargs
    ):
        """Filter indices to find those ready for consumption."""
        # Your filtering logic here
        ready_mask = torch.zeros(len(all_indices), dtype=torch.bool)
        # ... determine which indices are ready ...
        return [all_indices[i] for i in range(len(all_indices)) if ready_mask[i]]
```

## Examples

See the following files for complete examples:

- `recipe/samplers_simple_demo.py` - Standalone demonstration of samplers
- `tests/test_samplers.py` - Comprehensive test suite showing all use cases

## Design Philosophy

The sampler abstraction follows the Single Responsibility Principle:
- **Controller**: Manages metadata, production/consumption status
- **Sampler**: Defines consumption patterns and data selection logic
- **Client**: Handles data transfer and user interactions

This separation enables:
1. **Flexibility**: Easy to add new sampling strategies
2. **Reusability**: Samplers can be shared across different algorithms
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Changes to sampling logic don't affect controller logic

## API Reference

### BaseSampler

Abstract base class for all samplers.

**Methods:**
- `sample(ready_indices, batch_size, **kwargs) -> list[int]`: Select indices from ready pool
- `filter_ready_indices(all_indices, production_status, consumption_status, data_fields, field_mapping, **kwargs) -> list[int]`: Filter indices based on readiness criteria

### SequentialSampler

Default sampler with sequential selection.

**Parameters:** None

### GRPOSampler

Group-based sampler for GRPO algorithms.

**Parameters:**
- `num_n_samples (int)`: Number of samples per group

### DPSampler

Data parallel aware sampler.

**Parameters:**
- `dp_rank (int)`: Rank within DP group (0 to dp_size-1)
- `dp_size (int)`: Total number of DP groups
- `world_rank (int)`: Optional global rank (for logging)
- `world_size (int)`: Optional total processes (for validation)
