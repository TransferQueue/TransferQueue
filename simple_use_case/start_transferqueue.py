import logging
import math
import sys
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from transfer_queue.data_system import TransferQueueController, TransferQueueStorageSimpleUnit, process_zmq_server_info


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ray.init(runtime_env={"env_vars":{"RAY_DEBUG": "1", "RAY_DEDUP_LOGS":"0"}})


def initialize_data_system(config):
    # 1. 初始化TransferQueueStorage
    total_storage_size = (config.global_batch_size * config.num_global_batch)
    data_system_storage_units = {}
    for storage_unit_rank in range(config.num_data_storage_units):
        # TransferQueueStorage通过Ray拉起，是一个ray.remote修饰的类
        storage_node = TransferQueueStorageSimpleUnit.remote(
            storage_unit_id=storage_unit_rank,  # FIXME 临时回滚：后续StorageUnit不在主控指定ID
            storage_size=math.ceil(total_storage_size / config.num_data_storage_units)
        )
        data_system_storage_units[storage_unit_rank] = storage_node
        logger.info(f"TransferQueueStorageSimpleUnit #{storage_unit_rank} has been created.")

    # 2. 初始化TransferQueueController
    # 这里支持多controller实例以实现负载均衡，支持大规模扩展。不同controller可分配至不同RL计算任务
    data_system_controllers = {}
    for controller_rank in range(config.num_data_controllers):
        data_system_controllers[controller_rank] = TransferQueueController.remote(
            controller_id=str(controller_rank),  # FIXME 临时回滚：后续Controller不在主控指定ID
            num_storage_units=config.num_data_storage_units,
            global_batch_size=config.global_batch_size,
            num_global_batch=config.num_global_batch,
            num_n_samples=1,
        )
        logger.info(f"TransferQueueController #{controller_rank} has been created.")

    # 3. 将Controller注册至各个Storage
    # 每个Storage Unit拿到所有Controller的handler，通过Ray拿到对应的IP+端口，之后建立ZMQ Socket进行消息传输
    data_system_controller_infos = process_zmq_server_info(data_system_controllers)
    data_system_storage_unit_infos = process_zmq_server_info(data_system_storage_units)

    ray.get([storage_unit.register_controller_info.remote(data_system_controller_infos) for storage_unit in
             data_system_storage_units.values()])

    # 4. 创建Client
    from transfer_queue.data_system import TransferQueueClient
    data_system_client = TransferQueueClient(
        client_id='Trainer',
        controller_infos=data_system_controller_infos[0],  # TODO: 主控Client感知所有controller，WorkerGroup和Worker的Client感知一个controller
        storage_infos=data_system_storage_unit_infos,
        dp_world_size=None,
        num_dp_groups=None,
        dp_rank=None,
        rank_id=None
    )

    return data_system_controllers, data_system_storage_units, data_system_client


def task_1(data):
    return data + 1


def do_some_tasks(data_meta, data_system_client):
    # 1. 根据data_meta通过client从storage unit中拉取真实data
    data = data_system_client.get_data(data_meta)

    output = task_1(data["input_ids"])

    # 2. 修改data_meta，用于存放当前任务返回结果的元数据
    data_meta.set_target_column(["task_1_res"])
    output = TensorDict({"task_1_res": output}, batch_size=output.size())

    # 3. 将结果写回对应的storage unit
    data_system_client.put(data=output, metadata=data_meta)
    return data_meta


def main(config):

    # Data System：基于Ray拉起Controller以及Storage
    data_system_controllers, data_system_storage_units, data_system_client = initialize_data_system(config)

    import time
    time.sleep(3)

    input_ids = torch.tensor([1, 2, 3, 4])
    prompt_batch = TensorDict({"input_ids": input_ids}, batch_size=input_ids.size())

    data_system_client.put_prompts(data=prompt_batch, data_columns=["input_ids"], global_step=0, n_samples_per_prompt=1)
    logger.info("demo put prompts ok! ")
    time.sleep(3)

    prompt_meta = data_system_client.get_meta(
        data_columns=['input_ids'],
        experience_count=config.global_batch_size,
        global_step=0,
        dp_world_size=1,  # DP总数
        get_n_samples=False,
        task_name='task1',
        schedule_policy='random_strategy'
    )
    logger.info("demo get meta ok! ")
    new_data_meta = do_some_tasks(prompt_meta, data_system_client)
    print("demo done!")

    # TODO: clear data
    # 对于主控的client，通知所有controller进行数据状态清空，主控返回metadata；client再根据metadata通知所有storage unit清空
    # client选择一个主controller拿到metadata，其他的controller直接清空不用返回metadata即可
    # data_system_client.clear(global_step=0)


if __name__ == "__main__":
    config_str = """
      global_batch_size: 4
      num_global_batch: 1 
      num_data_storage_units: 1
      num_data_controllers: 1

    """
    dict_conf = OmegaConf.create(config_str)

    main(dict_conf)
