import asyncio
import logging
import math
import os
import sys
import time
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from tensordict.tensorclass import NonTensorData
import random
import numpy as np

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


from transfer_queue import (  # noqa: E402
    AsyncTransferQueueClient,
    BatchMeta,
    SimpleStorageUnit,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.utils.utils import get_placement_group  # noqa: E402


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Ray双机集群配置
HEAD_NODE_IP = "61.28.30.25"
WORKER_NODE_IP = "61.28.30.27"
# ray start --head --object-store-memory=100000000000
# ray start --address='61.28.30.25:6379' --object-store-memory=100000000000

# 统一配置
config_str = """
  global_batch_size: 1024
  seq_length: 8192
  field_num: 10
  num_global_batch: 1 
  num_data_storage_units: 8
  num_data_controllers: 1
"""
dict_conf = OmegaConf.create(config_str)


def create_complex_test_case(batch_size=None, seq_length=None, field_num=None):
    """
    创建测试数据，包含tensor和NonTensorData
    """
    # 计算数据大小
    tensor_field_size_bytes = batch_size * seq_length * 4
    tensor_field_size_gb = tensor_field_size_bytes / (1024**3)
    
    num_tensor_fields = (field_num + 1) // 2
    num_nontensor_fields = field_num // 2
    
    total_tensor_size_gb = tensor_field_size_gb * num_tensor_fields
    total_nontensor_size_gb = (batch_size * 1024 / (1024**3)) * num_nontensor_fields
    total_size_gb = total_tensor_size_gb + total_nontensor_size_gb
    
    logger.info(f"Total data size: {total_size_gb:.6f} GB")
    
    # 创建字段
    fields = {}
    
    for i in range(field_num):
        field_name = f"field_{i}"
        
        if i % 2 == 0:  # tensor字段
            tensor_data = torch.randn(batch_size, seq_length, dtype=torch.float32)
            fields[field_name] = tensor_data
        else:  # NonTensorData字段
            str_length = 1024
            non_tensor_data = [
                ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=str_length))
                for _ in range(batch_size)
            ]
            fields[field_name] = NonTensorData(data=non_tensor_data, batch_size=(batch_size,), device=None)
    
    # 创建TensorDict
    batch_size_tuple = (batch_size,)
    prompt_batch = TensorDict(
        fields,
        batch_size=batch_size_tuple,
        device=None,
    )

    return prompt_batch, total_size_gb


@ray.remote
class RemoteDataStoreObjStore:
    def __init__(self):
        pass
    
    def get_data(self, data_handler):
        start_get = time.time()
        data = ray.get(data_handler)
        end_get = time.time()

        get_time = end_get - start_get
        return get_time


@ray.remote
class RemoteDataStoreRemote:
    def __init__(self):
        self.stored_data = None
    
    def put_data(self, data):
        self.stored_data = data
    
    def get_data(self):
        return self.stored_data

    def clear_data(self):
        self.stored_data = None


class RayBandwidthTester:
    def __init__(self, config, test_mode="obj_store"):     
        self.config = config
        self.test_mode = test_mode
        
        if test_mode == "obj_store":
            RemoteDataStore = RemoteDataStoreObjStore
        else:
            RemoteDataStore = RemoteDataStoreRemote

        self.remote_store = RemoteDataStore.options(
            num_cpus=10,
            resources={f"node:{WORKER_NODE_IP}": 0.001}
        ).remote()
        
        logger.info(f"Remote data store created on worker node {WORKER_NODE_IP}")

    def run_bandwidth_test(self):
        # 构造数据
        start_create_data = time.time()
        test_data, total_data_size_gb = create_complex_test_case(
            batch_size=self.config.global_batch_size, 
            seq_length=self.config.seq_length,
            field_num=self.config.field_num
        )
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        if self.test_mode == "obj_store":
            self._run_obj_store_test(test_data, total_data_size_gb)
        else:
            self._run_remote_test(test_data, total_data_size_gb)

    def _run_obj_store_test(self, test_data, total_data_size_gb):
        # 测试PUT吞吐量
        start_time = time.time()
        data_handler = ray.put(test_data)
        ray.get(self.remote_store.get_data.remote([data_handler]))
        end_time = time.time()

        transfer_time = (end_time-start_time)

        # 计算吞吐量
        throughput = (total_data_size_gb * 8) / transfer_time
        
        # 输出汇总结果
        logger.info("="*60)
        logger.info("RAY OBJECT STORE BANDWIDTH TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Data Size: {(total_data_size_gb):.6f} GB")
        logger.info(f"Transfer Time: {transfer_time:.8f}s")
        logger.info(f"Throughput: {throughput:.8f} Gb/s")

    def _run_remote_test(self, test_data, total_data_size_gb):
        # 测试PUT吞吐量
        logger.info(f"Starting Ray PUT bandwidth test...")
        start_put = time.time()
        put_result = ray.get(self.remote_store.put_data.remote(test_data))
        end_put = time.time()
        put_time = end_put - start_put
        logger.info(f"PUT Time: {put_time:.8f}s")
        
        time.sleep(2)
        
        # 测试GET吞吐量
        logger.info(f"Starting Ray GET bandwidth test...")
        start_get = time.time()
        retrieved_data = ray.get(self.remote_store.get_data.remote())
        end_get = time.time()
        get_time = end_get - start_get
        logger.info(f"GET Time: {get_time:.8f}s")
        
        # 清理数据
        ray.get(self.remote_store.clear_data.remote())

        # 计算吞吐量
        put_throughput = (total_data_size_gb * 8) / put_time
        get_throughput = (total_data_size_gb * 8) / get_time
        
        # 输出汇总结果
        logger.info("="*60)
        logger.info("RAY REMOTE ACTOR BANDWIDTH TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput (Head->Worker): {put_throughput:.8f} Gb/s")
        logger.info(f"GET Throughput (Worker->Head): {get_throughput:.8f} Gb/s")
        logger.info(f"Round-trip Average Throughput: {total_data_size_gb * 16 / (put_time + get_time):.8f} Gb/s")


class TQBandwidthTester:
    def __init__(self, config, remote_mode=False):
        self.config = config
        self.remote_mode = remote_mode
        self.data_system_client = self._initialize_data_system()

    def _initialize_data_system(self):
        # 1. 初始化TransferQueueStorage
        total_storage_size = (self.config.global_batch_size * self.config.num_global_batch)
        self.data_system_storage_units = {}
        
        if self.remote_mode:
            # 限制在远程节点
            for storage_unit_rank in range(self.config.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    num_cpus=10,
                    resources={f"node:{WORKER_NODE_IP}": 0.001},
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units)
                )
                self.data_system_storage_units[storage_unit_rank] = storage_node
        else:
            storage_placement_group = get_placement_group(self.config.num_data_storage_units, num_cpus_per_actor=10)
            for storage_unit_rank in range(self.config.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    placement_group=storage_placement_group,
                    placement_group_bundle_index=storage_unit_rank,
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units)
                )
                self.data_system_storage_units[storage_unit_rank] = storage_node

        logger.info(f"TransferQueueStorageSimpleUnit #0 ~ #{storage_unit_rank} has been created.")

        # 2. Initialize TransferQueueController (single controller only)
        self.data_system_controller = TransferQueueController.remote()
        logger.info("TransferQueueController has been created.")

        # 3. 将Controller注册至各个Storage
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})  # Note: Need to generate a new DictConfig
        # with allow_objects=True to maintain ZMQServerInfo instance. Otherwise it will be flattened to dict
        tq_config.controller_info = self.data_system_controller_info
        tq_config.storage_unit_infos = self.data_system_storage_unit_infos
        self.config = OmegaConf.merge(tq_config, self.config)


        # 4. 创建Client
        self.data_system_client = AsyncTransferQueueClient(
            client_id='Trainer',
            controller_info=self.data_system_controller_info
        )
        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)
        return self.data_system_client

    def run_bandwidth_test(self):
        # 构造数据
        logger.info("Creating large batch for bandwidth test...")
        start_create_data = time.time()
        big_input_ids, total_data_size_gb = create_complex_test_case(
            batch_size=self.config.global_batch_size, 
            seq_length=self.config.seq_length,
            field_num=self.config.field_num
        )
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")
        
        # 测试PUT吞吐量
        logger.info(f"Starting PUT operation...")
        start_async_put = time.time()
        asyncio.run(self.data_system_client.async_put(data=big_input_ids, partition_id=f"train_0"))
        end_async_put = time.time()
        put_time = end_async_put - start_async_put
        
        put_throughput_gbps = (total_data_size_gb * 8) / put_time
        logger.info(f"async_put cost time: {put_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s")

        time.sleep(2)

        # 获取META
        logger.info("Starting GET_META operation...")
        start_async_get_meta = time.time()
        prompt_meta = asyncio.run(self.data_system_client.async_get_meta(
            data_fields=list(big_input_ids.keys()),
            batch_size=big_input_ids.size(0),
            partition_id=f"train_0",
            task_name='generate_sequences',
        ))
        end_async_get_meta = time.time()
        logger.info(f"async_get_meta cost time: {end_async_get_meta - start_async_get_meta:.8f}s")

        time.sleep(2)

        # 测试GET吞吐量
        logger.info(f"Starting GET_DATA operation...")
        start_async_get_data = time.time()
        data = asyncio.run(self.data_system_client.async_get_data(prompt_meta))
        end_async_get_data = time.time()
        get_time = end_async_get_data - start_async_get_data
        get_throughput_gbps = (total_data_size_gb * 8) / get_time
        
        logger.info(f"async_get_data cost time: {get_time:.8f}s")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s")
        
        # 输出汇总信息
        mode_name = "TQ REMOTE" if self.remote_mode else "TQ NORMAL"
        logger.info("="*60)
        logger.info(f"{mode_name} BANDWIDTH TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s")
        logger.info(f"Network Round-trip Throughput: {(total_data_size_gb * 16) / (put_time + get_time):.8f} Gb/s")


class TQMooncakeBandwidthTester:
    def __init__(self, config, mooncake_config=None):
        self.config = config
        self.mooncake_config = mooncake_config or {}
        self.data_system_client = self._initialize_data_system()

    def _initialize_data_system(self):
        # 1. Initialize TransferQueueController (single controller only)
        self.data_system_controller = TransferQueueController.remote()
        logger.info("TransferQueueController has been created.")

        # 2. 获取Controller信息
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)

        # 3. 配置Mooncake相关参数
        tq_config = OmegaConf.create({}, flags={"allow_objects": True})
        tq_config.controller_info = self.data_system_controller_info
        
        # Mooncake配置参数
        metadata_server = self.mooncake_config.get("metadata_server")
        master_server_address = self.mooncake_config.get("master_server_address")
        local_hostname = self.mooncake_config.get("local_hostname", "localhost")
        global_segment_size = self.mooncake_config.get("global_segment_size", 512 * 1024 * 1024)
        local_buffer_size = self.mooncake_config.get("local_buffer_size", 128 * 1024 * 1024)
        protocol = self.mooncake_config.get("protocol", "tcp")
        device_name = self.mooncake_config.get("device_name", "")

        if metadata_server is None:
            raise ValueError("Missing 'metadata_server' in mooncake_config")
        if master_server_address is None:
            raise ValueError("Missing 'master_server_address' in mooncake_config")

        tq_config.metadata_server = metadata_server
        tq_config.master_server_address = master_server_address
        tq_config.local_hostname = local_hostname
        tq_config.global_segment_size = global_segment_size
        tq_config.local_buffer_size = local_buffer_size
        tq_config.protocol = protocol
        tq_config.device_name = device_name
        tq_config.client_name = "MooncakeStorageClient"
        
        self.config = OmegaConf.merge(tq_config, self.config)

        # 4. 创建Client
        self.data_system_client = AsyncTransferQueueClient(
            client_id='Trainer',
            controller_info=self.data_system_controller_info
        )
        self.data_system_client.initialize_storage_manager(manager_type="MooncakeStorageManager", config=self.config)
        logger.info("TQMooncakeBandwidthTester initialized with MooncakeStorageManager")
        return self.data_system_client

    def run_bandwidth_test(self):
        # 构造数据
        logger.info("Creating large batch for bandwidth test...")
        start_create_data = time.time()
        big_input_ids, total_data_size_gb = create_complex_test_case(
            batch_size=self.config.global_batch_size, 
            seq_length=self.config.seq_length,
            field_num=self.config.field_num
        )
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")
        
        # 测试PUT吞吐量
        logger.info(f"Starting PUT operation...")
        start_async_put = time.time()
        asyncio.run(self.data_system_client.async_put(data=big_input_ids, partition_id=f"train_0"))
        end_async_put = time.time()
        put_time = end_async_put - start_async_put
        
        put_throughput_gbps = (total_data_size_gb * 8) / put_time
        logger.info(f"async_put cost time: {put_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s")

        time.sleep(2)

        # 获取META
        logger.info("Starting GET_META operation...")
        start_async_get_meta = time.time()
        prompt_meta = asyncio.run(self.data_system_client.async_get_meta(
            data_fields=list(big_input_ids.keys()),
            batch_size=big_input_ids.size(0),
            partition_id=f"train_0",
            task_name='generate_sequences',
        ))
        end_async_get_meta = time.time()
        logger.info(f"async_get_meta cost time: {end_async_get_meta - start_async_get_meta:.8f}s")

        time.sleep(2)

        # 测试GET吞吐量
        logger.info(f"Starting GET_DATA operation...")
        start_async_get_data = time.time()
        data = asyncio.run(self.data_system_client.async_get_data(prompt_meta))
        end_async_get_data = time.time()
        get_time = end_async_get_data - start_async_get_data
        get_throughput_gbps = (total_data_size_gb * 8) / get_time
        
        logger.info(f"async_get_data cost time: {get_time:.8f}s")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s")
        
        # 输出汇总信息
        logger.info("="*60)
        logger.info("TQ MOONCAKE BANDWIDTH TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s")
        logger.info(f"Network Round-trip Throughput: {(total_data_size_gb * 16) / (put_time + get_time):.8f} Gb/s")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python performance_test.py <test_mode> [mooncake_config_file]")
        print("Available test modes:")
        print("  ray-obj-store    - Ray Object Store bandwidth test")
        print("  ray-remote       - Ray Remote Actor bandwidth test") 
        print("  tq-normal        - TQ Normal mode bandwidth test")
        print("  tq-remote        - TQ Remote mode bandwidth test")
        print("  tq-mooncake      - TQ Mooncake backend bandwidth test")
        print("")
        print("For tq-mooncake mode, you can optionally provide a config file with mooncake parameters:")
        print("  metadata_server: <server_address>")
        print("  master_server_address: <server_address>")
        print("  local_hostname: <hostname> (optional, default: localhost)")
        print("  protocol: <tcp|rdma> (optional, default: tcp)")
        return
    
    test_mode = sys.argv[1]
    
    if test_mode == "ray-obj-store":
        logger.info(f"Starting Ray Object Store bandwidth test")
        tester = RayBandwidthTester(config=dict_conf, test_mode="obj_store")
        tester.run_bandwidth_test()
        logger.info("Ray Object Store bandwidth test completed successfully!")
    
    elif test_mode == "ray-remote":
        logger.info(f"Starting Ray Remote Actor bandwidth test")
        tester = RayBandwidthTester(config=dict_conf, test_mode="remote")
        tester.run_bandwidth_test()
        logger.info("Ray Remote Actor bandwidth test completed successfully!")
    
    elif test_mode in ["tq-normal", "tq-remote"]:
        remote_mode = (test_mode == "tq-remote")
        mode_name = "TQ Remote" if remote_mode else "TQ Normal"
        logger.info(f"Starting {mode_name} bandwidth test")
        
        tester = TQBandwidthTester(config=dict_conf, remote_mode=remote_mode)
        tester.run_bandwidth_test()
        logger.info(f"{mode_name} bandwidth test completed successfully!")
    
    elif test_mode == "tq-mooncake":
        logger.info("Starting TQ Mooncake bandwidth test")
        
        mooncake_config = {}
        if len(sys.argv) >= 3:
            config_file = sys.argv[2]
            if os.path.exists(config_file):
                mooncake_config = OmegaConf.load(config_file)
                logger.info(f"Loaded mooncake config from {config_file}")
            else:
                logger.warning(f"Config file {config_file} not found, using default/empty config")
        else:
            logger.warning("No mooncake config file provided, using default/empty config")
            logger.warning("Please ensure mooncake_config is set via environment variables or defaults")
        
        tester = TQMooncakeBandwidthTester(config=dict_conf, mooncake_config=mooncake_config)
        tester.run_bandwidth_test()
        logger.info("TQ Mooncake bandwidth test completed successfully!")
    
    else:
        print(f"Unknown test mode: {test_mode}")
        print("Available test modes: ray-obj-store, ray-remote, tq-normal, tq-remote, tq-mooncake")


if __name__ == "__main__":
    main()