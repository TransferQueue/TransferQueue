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

from enum import Enum

import ray
import torch


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class TransferQueueRole(ExplicitEnum):
    CONTROLLER = "TransferQueueController"
    STORAGE = "TransferQueueStorage"
    CLIENT = "TransferQueueClient"


# production_status enum: 0: not produced, 1: ready for consume, 2: consumed
class ProductionStatus(ExplicitEnum):
    NOT_PRODUCED = 0
    READY_FOR_CONSUME = 1
    CONSUMED = 2


def get_placement_group(num_ray_actors: int, num_cpus_per_actor: int = 1):
    """
    Create a placement group with SPREAD strategy for Ray actors.

    Args:
        num_ray_actors (int): Number of Ray actors to create.
        num_cpus_per_actor (int): Number of CPUs to allocate per actor.

    Returns:
        placement_group: The created placement group.
    """
    bundle = {"CPU": num_cpus_per_actor}
    placement_group = ray.util.placement_group([bundle for _ in range(num_ray_actors)], strategy="SPREAD")
    ray.get(placement_group.ready())
    return placement_group


def sequential_sampler(
    ready_for_consume_idx: list[int],
    batch_size: int,
    get_n_samples: bool,
    n_samples_per_prompt: int,
) -> list[int]:
    """
    Sequentially samples a batch of indices from global indexes ready_for_consume_idx.

    Args:
        ready_for_consume_idx: A sorted list of available indices for sampling.
            - When get_n_samples=True:
                Expected to be grouped by prompts, e.g.,
                [0,1,2,3, 8,9,10,11, 12,13,14,15] (3 groups of 4 samples each)
            - When get_n_samples=False:
                Can be any ordered list, e.g., [0,3,5,6,7,8]
        batch_size: Total number of samples to return
        get_n_samples: Flag indicating the sampling mode
        n_samples_per_prompt: Number of samples per prompt (used when get_n_samples=True)

    Returns:
        list[int]: Sequentially sampled indices of length batch_size
    """
    if get_n_samples:
        assert len(ready_for_consume_idx) % n_samples_per_prompt == 0
        assert batch_size % n_samples_per_prompt == 0
        batch_size_n_samples = batch_size // n_samples_per_prompt

        group_ready_for_consume_idx = torch.tensor(ready_for_consume_idx, dtype=torch.int).view(
            -1, n_samples_per_prompt
        )

        sampled_indexes = group_ready_for_consume_idx[list(range(batch_size_n_samples))].flatten().tolist()
    else:
        sampled_indexes = [int(ready_for_consume_idx[i]) for i in range(batch_size)]
    return sampled_indexes
