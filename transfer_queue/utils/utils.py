from enum import Enum

import ray
import torch
from tensordict import TensorDict


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


def random_sampler(
    ready_for_consume_idx: list[int],
    batch_size: int,
    get_n_samples: bool,
    n_samples_per_prompt: int,
) -> list[int]:
    """
    random sampling batch_size samples from global indexes ready_for_consume_idx
    input example:
        if get_n_samples: (group_num=3, group_size=4)
            ready_for_consume_idx could look like: [0, 1, 2, 3,   8, 9, 10, 11,   16, 17, 18, 19]
        else:
            ready_for_consume_idx could look like: [2, 5, 6]
    """
    if get_n_samples:
        assert len(ready_for_consume_idx) % n_samples_per_prompt == 0
        assert batch_size % n_samples_per_prompt == 0
        batch_size_n_samples = batch_size // n_samples_per_prompt

        group_ready_for_consume_idx = torch.tensor(ready_for_consume_idx, dtype=torch.int).view(
            -1, n_samples_per_prompt
        )

        weights = torch.ones(group_ready_for_consume_idx.size(0))
        sampled_indexes_idx = torch.multinomial(weights, batch_size_n_samples, replacement=False).tolist()
        sampled_indexes = group_ready_for_consume_idx[sampled_indexes_idx].flatten().tolist()
    else:
        weights = torch.ones(len(ready_for_consume_idx))
        sampled_indexes_idx = torch.multinomial(weights, batch_size, replacement=False).tolist()
        sampled_indexes = [int(ready_for_consume_idx[i]) for i in sampled_indexes_idx]
    return sampled_indexes


def extract_field_info(tensor_dict: TensorDict, set_all_ready: bool = True) -> dict:
    """
    Extract field information from a TensorDict. If data in tensor_dict does not have dtype or shape attribute,
    the corresponding dtype or shape will be set to None.

    Args:
        tensor_dict (TensorDict): The input TensorDict.
        set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME. Default is True.

    Returns:
        dict: A dictionary containing field names, dtypes, shapes, and production_status. Example:
            {
                "names": ["field1", "field2"],
                "dtypes": [
                    {"field1": torch.float32, "field2": torch.int64},
                    {"field1": torch.float32, "field2": torch.int64}
                ],
                "shapes": [
                    {"field1": torch.Size([10]), "field2": torch.Size([5, 5])},
                    {"field1": torch.Size([10]), "field2": torch.Size([5, 5])}
                ],
                "production_status": [
                    {"field1": ProductionStatus.READY_FOR_CONSUME, "field2": ProductionStatus.READY_FOR_CONSUME},
                    {"field1": ProductionStatus.READY_FOR_CONSUME, "field2": ProductionStatus.READY_FOR_CONSUME}
                ]
            }
    """
    field_info: dict[str, list] = {"names": [], "dtypes": [], "shapes": [], "production_status": []}
    batch_size = tensor_dict.batch_size[0]
    field_info["names"] = list(tensor_dict.keys())

    for sample in range(batch_size):
        field_info["dtypes"].append({})
        field_info["shapes"].append({})
        field_info["production_status"].append({})
        for field_name in field_info["names"]:
            value = tensor_dict[field_name][sample]
            field_info["dtypes"][sample][field_name] = value.dtype if hasattr(value, "dtype") else None
            field_info["shapes"][sample][field_name] = value.shape if hasattr(value, "shape") else None
            field_info["production_status"][sample][field_name] = (
                ProductionStatus.READY_FOR_CONSUME if set_all_ready else ProductionStatus.NOT_PRODUCED
            )
    return field_info
