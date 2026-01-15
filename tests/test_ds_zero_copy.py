import sys
import time
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.client import TransferQueueClient  # noqa: E402
from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402
from transfer_queue.storage.managers.yuanrong_manager import YuanrongStorageManager  # noqa: E402
from transfer_queue.storage.clients.yuanrong_client import YuanrongStorageClient  # noqa: E402
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory  # noqa: E402
from transfer_queue.utils.zmq_utils import ZMQServerInfo  # noqa: E402
from transfer_queue import (
    TransferQueueController,
    process_zmq_server_info,    
)

def tensordict_memory_mb(td):
    total_bytes = sum(tensor.element_size() * tensor.numel() for tensor in td.values())
    return total_bytes / (1024 * 1024)

@ray.remote
class WriterActor:
    def __init__(self, controller_info, config):

        self.client = TransferQueueClient(client_id=f"writer_{id(self)}", controller_info=controller_info)
        self.client.initialize_storage_manager("YuanrongStorageManager", config)
        self.data = None

    def generate_data(
        self, partition_id, batch_size: int = 10000, seq_len: int = 10000
    ) 
        data = TensorDict(
            {
                "input_ids": torch.randn(batch_size, seq_len, dtype=torch.float32),
            },
            batch_size=batch_size,
        )

        size = tensordict_memory_mb(data)
        print(f"Generated data of size {size:.2f} MB")
        self.data = data

    def put_once(self, partition_id)
        t0 = time.perf_counter()
        batch_meta = self.client.put(data=self.data, partition_id=partition_id)
        return time.perf_counter() - t0, batch_meta


@ray.remote
class ReaderActor:
    def __init__(self, controller_info, config):

        self.client = TransferQueueClient(client_id=f"reader_{id(self)}", controller_info=controller_info)
        self.client.initialize_storage_manager("YuanrongStorageManager", config)

    def get_once(self, metadata: BatchMeta):
        t0 = time.perf_counter()
        self.client.get_data(metadata)
        return time.perf_counter() - t0

def main():
    if not ray.is_initialized():
        ray.init(address="auto")

    data_system_controller = TransferQueueController.remote()
    controller_info = process_zmq_server_info(data_system_controller)

    config_writer = {
        "client_name": "YuanrongStorageClient",
        "controller_info": controller_info,
        "host": "10.90.41.116",
        "port": 36666,
    }

    config_reader = {
        "client_name": "YuanrongStorageClient",
        "controller_info": controller_info,
        "host": "10.90.41.117",
        "port": 36666,
    }

    nodes = ray.nodes()
    ip_to_nodeid = {}
    for n in nodes:
        addr = n.get("NodeManagerAddress") or n.get("node_ip_address") or n.get("NodeIP")
        node_id = n["NodeID"] if "NodeID" in n else n.get("NodeID") or n.get("node_id")
        if addr and node_id:
            ip_to_nodeid[addr] = node_id

    ip_A = "10.90.41.117"  # Writer
    ip_B = "10.90.41.116"  # Reader
    node_id_A = ip_to_nodeid.get(ip_A)
    node_id_B = ip_to_nodeid.get(ip_B)
    assert node_id_A and node_id_B, f"cannot find node ids for {ip_A}, {ip_B}: {ip_to_nodeid}"

    writer = WriterActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id_A, soft=False),
    ).remote(controller_info, config_writer)
    reader = ReaderActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id_B, soft=False),
    ).remote(controller_info, config_reader)

    batch_metas = []
    put_times = []
    get_times =[]

    for i in range(3):
        partition_id = f"train_step_{i}"
        ray.get(writer.generate_data.remote(batch_size=512, seq_len=32 * 1024))
        cost, batch_meta = ray.get(writer.put_once.remote(partition_id))
        print(f"[WriterActor] The time consumed by the {i}th put costs: {cost:.2f}s")
        batch_metas.append(batch_meta)
        put_times.append(cost)

    for i, meta in enumerate(batch_metas):
        cost = ray.get(reader.get_once.remote(meta))
        get_times.append(cost)
        print(f"[ReaderActor] The time consumed by the {i}th get costs: {cost:.2f}s")

    avg_put_time = sum(put_times) / len(put_times)
    avg_get_time = sum(get_times) / len(get_times)
    print(f"Average put time: {avg_put_time:.2f}s")
    print(f"Average get time: {avg_get_time:.2f}s")

if __name__ == "__main__":
    main()
