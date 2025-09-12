import asyncio
import logging
import math
import sys
import time
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent.parent
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
            storage_size=math.ceil(total_storage_size / config.num_data_storage_units)
        )
        data_system_storage_units[storage_unit_rank] = storage_node
        logger.info(f"TransferQueueStorageSimpleUnit #{storage_unit_rank} has been created.")

    # 2. 初始化TransferQueueController
    # 这里支持多controller实例以实现负载均衡，支持大规模扩展。不同controller可分配至不同RL计算任务
    data_system_controllers = {}
    for controller_rank in range(config.num_data_controllers):
        data_system_controllers[controller_rank] = TransferQueueController.remote(
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
        storage_infos=data_system_storage_unit_infos
    )

    return data_system_controllers, data_system_storage_units, data_system_client


def generate_sequences(data):
    time.sleep(3)
    return data


def compute_old_log_prob(data1, data2):
    time.sleep(3)
    return data1


def actor_rollout_wg_generate_sequences(data_meta, data_system_client):
    # 1. 根据data_meta通过client从storage unit中拉取真实data
    data = data_system_client.get_data(data_meta)
    logger.info(f"demo get data {data}")

    output = generate_sequences(data["input_ids"])

    # 2. 修改data_meta，用于存放当前任务返回结果的元数据
    data_meta.set_output_fields(["generate_sequences_ids"])
    output = TensorDict({"generate_sequences_ids": output}, batch_size=output.size(0))

    # 3. 根据data_meta将结果写回storage unit
    data_system_client.put(data=output, metadata=data_meta)
    logger.info("demo put data to storages done")

    return data_meta


def actor_rollout_wg_compute_old_log_prob(data_meta, data_system_client):
    # 1. 根据data_meta通过client从storage unit中拉取真实data
    data = data_system_client.get_data(data_meta)
    logger.info(f"demo get data {data}")

    output = compute_old_log_prob(data["input_ids"], data["generate_sequences_ids"])

    # 2. 修改data_meta，用于存放当前任务返回结果的元数据
    data_meta.set_output_fields(["old_log_prob"])
    output = TensorDict({"old_log_prob": output}, batch_size=output.size(0))

    # 3. 根据data_meta将结果写回storage unit
    data_system_client.put(data=output, metadata=data_meta)
    logger.info("demo put data to storages done")

    return data_meta


# Simulate the fit function of the trainer
def fit(config, data_system_client):
    for epoch in range(2):
        train_dataloader = 2
        for step in range(train_dataloader):
            input_ids = (torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])) * (step + 1)
            prompt_batch = TensorDict({"input_ids": input_ids}, batch_size=input_ids.size(0))

            data_system_client.put(data=prompt_batch, data_fields=["input_ids"], global_step=step)
            logger.info("demo put prompts ok! ")
            time.sleep(5)

            prompt_meta = data_system_client.get_meta(
                data_fields=['input_ids'],
                batch_size=config.global_batch_size,
                global_step=step,
                get_n_samples=False,
                task_name='generate_sequences',
            )
            # Set output fields for RL training - in this case, we want to generate sequences from input_ids
            logger.info(f"demo get meta {prompt_meta}")

            # Simulate calling the generate sequences task of the worker group
            actor_rollout_wg_generate_sequences(prompt_meta, data_system_client)

            log_prob_meta = data_system_client.get_meta(
                data_fields=['input_ids', 'generate_sequences_ids'],
                batch_size=config.global_batch_size,
                global_step=0,
                get_n_samples=False,
                task_name='compute_old_log_prob',
            )
            # Set output fields for RL training - we want to compute log probs for the generated sequences
            logger.info(f"demo get log prob meta: {log_prob_meta}")

            # Simulate calling the compute old log prob task of the worker group
            actor_rollout_wg_compute_old_log_prob(log_prob_meta, data_system_client)

            # 对于主控的client，通知所有controller进行数据状态清空，主控返回metadata；client再根据metadata通知所有storage unit清空
            # client选择一个主controller拿到metadata，其他的controller直接清空不用返回metadata即可
            data_system_client.clear(global_step=step)
            logger.info("clear ok! ")
    logger.info("demo done!")


def main(config):
    # Initialize Data System：基于Ray拉起Controller以及Storage
    data_system_controllers, data_system_storage_units, data_system_client = initialize_data_system(config)
    import time
    time.sleep(5)

    fit(config, data_system_client)


if __name__ == "__main__":
    config_str = """
      global_batch_size: 4
      num_global_batch: 1 
      num_data_storage_units: 2
      num_data_controllers: 1

    """
    dict_conf = OmegaConf.create(config_str)

    main(dict_conf)

