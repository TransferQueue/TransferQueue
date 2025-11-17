from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import asyncio
import ray
import os
import torch
from typing import Any
from tensordict import TensorDict, NonTensorData

from transfer_queue.metadata import BatchMeta, SampleMeta, FieldMeta
from transfer_queue.utils.zmq_utils import ZMQServerInfo
from transfer_queue.storage.clients.ray_storage_client import RayStorageClient
from transfer_queue.storage.managers.ray_kv_manager import RayKVStorageManager
from transfer_queue.client import AsyncTransferQueueClient
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory

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
def ensure_mock_storage_manager_registered():
    """Ensure MockRayKVStorageManager is registered in current process."""
    from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory
    from transfer_queue.storage.managers.ray_kv_manager import RayKVStorageManager

    if "RAY_MOCK" not in TransferQueueStorageManagerFactory._registry:
        @TransferQueueStorageManagerFactory.register("RAY_MOCK")
        class MockRayKVStorageManager(RayKVStorageManager):
            def _connect_to_controller(self): pass
            def _do_handshake_with_controller(self): pass
            async def notify_data_update(*args, **kwargs): return
        print("Registered RAY_MOCK in current process")

ensure_mock_storage_manager_registered()

# Step 3: Define Writer and Reader Actors
@ray.remote(num_gpus=1)
class WriterActor:
    def __init__(self, controller_info, config):
        ensure_mock_storage_manager_registered()
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        self.client = AsyncTransferQueueClient(
            client_id=f"writer_{id(self)}",
            controller_info=controller_info
        )
        self.client.initialize_storage_manager("RAY_MOCK", config)

    async def write_data(self, partition_id: str):
        batch_size = 3
        seq_len = 5

        data = TensorDict({
            "input_ids": torch.randint(1, 100, (batch_size, seq_len)).cuda(),
            "labels": torch.randn(batch_size, 2).cuda(),
            "nested_tensor": torch.nested.as_nested_tensor([
                torch.randn(torch.randint(2, 5, ()).item(), 3).cuda() for _ in range(batch_size)
            ]),
        }, batch_size=[batch_size])

        samples = [
            SampleMeta(
                global_index=i,
                partition_id=partition_id,
                fields={
                    "input_ids": FieldMeta(name="input_ids", dtype=torch.int64, shape=(seq_len,)),
                    "labels": FieldMeta(name="labels", dtype=torch.float32, shape=(2,)),
                    "nested_tensor": FieldMeta(name="nested_tensor", dtype=torch.Tensor, shape=()),
                }
            ) for i in range(batch_size)
        ]
        metadata = BatchMeta(samples=samples)

        print(f"[WriterActor] Writing data for partition {partition_id}...")
        await self.client.async_put(data=data, metadata=metadata)
        print("[WriterActor] Write completed.")

        return metadata


@ray.remote(num_gpus=1)
class ReaderActor:
    def __init__(self, controller_info, config):
        ensure_mock_storage_manager_registered()
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        self.client = AsyncTransferQueueClient(
            client_id=f"reader_{id(self)}",
            controller_info=controller_info
        )
        self.client.initialize_storage_manager("RAY_MOCK", config)

    async def read_data(self, metadata: BatchMeta):
        print(f"[ReaderActor] Reading {len(metadata)} samples...")
        result = await self.client.async_get_data(metadata)
        print("[ReaderActor] Read completed.")
        return result

# Step 4: Main Test Function
async def main():
    if not ray.is_initialized():
        ray.init(runtime_env={"working_dir": "."})

    client = None
    try:
        controller_info = create_mock_controller()
        config = {
            "client_name": "RAY",
            "controller_info": controller_info,
        }

        client = AsyncTransferQueueClient(
            client_id="test_driver",
            controller_info=controller_info
        )
        client.initialize_storage_manager("RAY_MOCK", config)

        print("Driver initialized (mocked)")

        writer = WriterActor.remote(controller_info, config)
        reader = ReaderActor.remote(controller_info, config)

        partition_id = "train_step_0"

        metadata = await writer.write_data.remote(partition_id)

        result = await reader.read_data.remote(metadata)

        print("Validating actor-to-actor transfer...")

        expected_input_ids = torch.randint(1, 100, (3, 5))
        assert result["input_ids"].shape == (3, 5), "Shape mismatch"
        assert result["labels"].shape == (3, 2), "Shape mismatch"
        assert len(result["nested_tensor"].unbind()) == 3, "nested tensor component count mismatch"

        print("Actor-to-Actor communication works!")

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