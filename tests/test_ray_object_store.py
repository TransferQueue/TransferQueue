import asyncio
from typing import Any

import ray
import torch
from tensordict import NonTensorData, TensorDict

from transfer_queue.client import AsyncTransferQueueClient
from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory
from transfer_queue.storage.managers.kv_manager import KVStorageManager
from transfer_queue.utils.zmq_utils import ZMQServerInfo

# Step 1: Mock Controller Role
try:
    from transfer_queue.role import TransferQueueRole
except ImportError:
    from enum import Enum

    class TransferQueueRole(Enum):
        CONTROLLER = "controller"
        STORAGE = "storage"


def create_mock_controller():
    return ZMQServerInfo(
        role=TransferQueueRole.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={
            "request_handle_socket": 9981,
            "data_status_update_socket": 9982,
            "handshake_socket": 9983,
        },
    )


# Step 2: Mock Storage Manager (Skip Controller Connect)
@TransferQueueStorageManagerFactory.register("KV_MOCK")
class MockKVStorageManager(KVStorageManager):
    def _connect_to_controller(self):
        pass

    def _do_handshake_with_controller(self):
        pass

    async def notify_data_update(*args, **kwargs):
        return


def _new_generate_values(data: TensorDict) -> list[Any]:
    values = []
    for field in sorted(data.keys()):
        col = data[field]
        if hasattr(col, "__len__"):
            for item in col:
                values.append(item)
        else:
            values.append(col)
    return values


KVStorageManager._generate_values = staticmethod(_new_generate_values)


def _patched_merge_tensors_to_tensordict(metadata: BatchMeta, values: list) -> TensorDict:
    """
    Patched version of _merge_tensors_to_tensordict that supports NonTensorData.
    """
    global_indexes = metadata.global_indexes
    field_names = sorted(metadata.field_names)
    expected_length = len(global_indexes) * len(field_names)

    if len(values) != expected_length:
        raise ValueError(f"Length of values ({len(values)}) does not match expected ({expected_length})")

    if len(values) == 0:
        return TensorDict({}, batch_size=len(global_indexes))

    # Grouping: Each field collects the corresponding data
    merged_data: dict[str, list] = {field: [] for field in field_names}

    value_idx = 0
    for field in field_names:
        for _ in range(len(global_indexes)):
            merged_data[field].append(values[value_idx])
            value_idx += 1

    tensor_data = {}
    for field, items in merged_data.items():
        # if the first element is str/list/tuple/dict，package it into NonTensorData then stack
        if isinstance(items[0], str | list | tuple | dict):
            ntd_items = [NonTensorData(x) for x in items]
            tensor_data[field] = torch.stack(ntd_items)
        else:
            try:
                tensor_data[field] = torch.stack(items)
            except RuntimeError:
                tensor_data[field] = torch.nested.as_nested_tensor(items)

    return TensorDict(tensor_data, batch_size=len(global_indexes))


KVStorageManager._merge_tensors_to_tensordict = staticmethod(_patched_merge_tensors_to_tensordict)


# Step 3: Main Test Function
async def main():
    if not ray.is_initialized():
        ray.init()

    client = None
    try:
        controller_info = create_mock_controller()
        client = AsyncTransferQueueClient(client_id="test_client", controller_info=controller_info)

        config = {
            "client_name": "RAY",
            "controller_info": controller_info,
        }

        client.initialize_storage_manager("KV_MOCK", config)
        print("Storage manager initialized (mocked controller)")

        # Step 4: Create Data & Metadata with Non-Tensor Fields
        batch_size = 3
        seq_len = 5

        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        labels = torch.randn(batch_size, 2)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        info_scalar = torch.zeros(batch_size, 1)

        # Variable-length nested tensors (each sample has a different length)
        nested_tensors = [torch.randn(torch.randint(2, 6, ()).item(), 3) for _ in range(batch_size)]
        nested_tensor = torch.nested.as_nested_tensor(nested_tensors)

        texts = ["query: what is RLHF?", "query: explain LoRA", "query: how to train LLM?"]
        tags_list = [["nlp", "rl"], ["lora", "peft"], ["training", "scaling"]]
        profiles = [
            {"level": "expert", "domain": "AI"},
            {"level": "beginner", "domain": "ML"},
            {"level": "intermediate", "domain": "CV"},
        ]
        image_shapes = [(224, 224, 3), (112, 112, 3), (512, 512, 3)]

        data = TensorDict(
            {
                "input_ids": input_ids,
                "labels": labels,
                "mask": mask,
                "info": info_scalar,
                "nested_tensor": nested_tensor,
                "text": torch.stack([NonTensorData(t) for t in texts]),
                "tags": torch.stack([NonTensorData(lst) for lst in tags_list]),
                "user_profile": torch.stack([NonTensorData(p) for p in profiles]),
                "image_shape": torch.stack([NonTensorData(shape) for shape in image_shapes]),
            },
            batch_size=[batch_size],
        )

        samples = []
        for i in range(batch_size):
            fields = {
                # tensor
                "input_ids": FieldMeta(name="input_ids", dtype=torch.int64, shape=(seq_len,)),
                "labels": FieldMeta(name="labels", dtype=torch.float32, shape=(2,)),
                "mask": FieldMeta(name="mask", dtype=torch.bool, shape=(seq_len,)),
                "info": FieldMeta(name="info", dtype=torch.float32, shape=(1,)),  # 实际是 float
                "nested_tensor": FieldMeta(name="nested_tensor", dtype=torch.Tensor, shape=()),
                # non-tensor
                "text": FieldMeta(name="text", dtype=str, shape=()),
                "tags": FieldMeta(name="tags", dtype=list, shape=()),
                "user_profile": FieldMeta(name="user_profile", dtype=dict, shape=()),
                "image_shape": FieldMeta(name="image_shape", dtype=tuple, shape=()),
            }
            sample = SampleMeta(global_index=i, partition_id="unified_test_partition", fields=fields)
            samples.append(sample)

        metadata = BatchMeta(samples=samples)
        print(f"Detected field names: {metadata.field_names}")

        # Step 5: Put Data
        print("Writing data via async_put...")
        await client.async_put(data=data, metadata=metadata)
        print("Data written successfully")

        # Step 6: Get Data
        print("Reading data via async_get_data...")
        result = await client.async_get_data(metadata)
        print("Data read successfully")

        # Step 7: Validate Results
        print("Validating results...")

        assert torch.equal(result["input_ids"], data["input_ids"])
        assert torch.allclose(result["labels"], data["labels"])
        assert torch.equal(result["mask"], data["mask"])
        assert result["nested_tensor"][0].shape == data["nested_tensor"][0].shape

        assert result["text"][0] == "query: what is RLHF?"
        assert result["tags"][1] == ["lora", "peft"]
        assert result["user_profile"][2]["domain"] == "CV"
        assert result["image_shape"][1] == (112, 112, 3)

        print("All non-tensor tests passed! Ray storage chain works on CPU.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if client is not None:
            client.close()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
