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
from tensordict import NonTensorData, TensorDict

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"
ray.init()


def compute_old_log_prob(data1, data2):
    time.sleep(3)
    return data1


def generate_sequences(data):
    time.sleep(3)
    return data


class ActorRolloutRefWorker:
    def actor_rollout_wg_generate_sequences(self, data_meta, data_system_client):
        # 1. 根据data_meta通过client从storage unit中拉取真实data
        data = asyncio.run(data_system_client.async_get_data(data_meta))
        logger.info(f"demo get data->generate_sequences {data}")

        output = generate_sequences(data["input_ids"])

        output = TensorDict(
            {
                "generate_sequences_ids": output,
                "non_tensor_data": torch.stack([NonTensorData("test_str") for _ in range(output.size(0))]),
                "nested_tensor": torch.nested.as_nested_tensor([torch.randn(1, 2) for _ in range(output.size(0))]),
            },
            batch_size=output.size(0),
        )

        # 2. 根据data_meta将结果写回storage unit
        asyncio.run(data_system_client.async_put(data=output, metadata=data_meta))
        data_meta.add_fields(output)
        logger.info("demo put data to storages done")

        return data_meta

    def actor_rollout_wg_compute_old_log_prob(self, data_meta, data_system_client):
        # 1. 根据data_meta通过client从storage unit中拉取真实data
        data = asyncio.run(data_system_client.async_get_data(data_meta))
        logger.info(f"demo get data->old_log_prob {data}")

        output = compute_old_log_prob(data["input_ids"], data["generate_sequences_ids"])

        output = TensorDict({"old_log_prob": output}, batch_size=output.size(0))

        # 2. 根据data_meta将结果写回storage unit
        asyncio.run(data_system_client.async_put(data=output, metadata=data_meta))
        data_meta.add_fields(output)
        logger.info("demo put data to storages done")

        return data_meta


@ray.remote
class AsyncvLLMServer:
    def __init__(self, config, data_system_controller_infos):
        self.config = config
        self.data_system_client = AsyncTransferQueueClient(
            client_id="AsyncvLLMServer",
            controller_infos=data_system_controller_infos[0],
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)

    async def generate(self, data_meta):
        data = await self.data_system_client.async_get_data(data_meta)
        logger.info(f"demo get data->generate_sequences {data}")

        data = data["input_ids"]
        data += 1
        await asyncio.sleep(3)

        output = TensorDict(
            {
                "generate_sequences_ids": data,
                "non_tensor_data": torch.stack([NonTensorData("test_str") for _ in range(data.size(0))]),
                "nested_tensor": torch.nested.as_nested_tensor([torch.randn(1, 2) for _ in range(data.size(0))]),
            },
            batch_size=data.size(0),
        )

        await self.data_system_client.async_put(data=output, metadata=data_meta)
        logger.info("demo Async Server put data to storages done")

        return data_meta


@ray.remote(num_cpus=1)
class AsyncRolloutWorker:
    def __init__(
        self,
        config,
        data_system_controller_infos,
    ):
        self.async_vllm_server = AsyncvLLMServer.remote(
            config,
            data_system_controller_infos,
        )

    async def generate_sequences(self, data_meta_chunk):
        tasks = []
        for i in range(data_meta_chunk.size):
            # asyncio.create_task cannot directly call Ray Actor methods,
            # otherwise an error will be reported：a coroutine was expected, got ObjectRef(xxx)
            tasks.append(asyncio.create_task(self.generate(data_meta_chunk[i])))
        data_metas = await asyncio.gather(*tasks)
        return BatchMeta.concat(data_metas)

    async def generate(self, data_meta):
        data_meta_new = await self.async_vllm_server.generate.remote(data_meta)
        return data_meta_new


class RolloutManager:
    def __init__(self, config, data_system_storage_unit_infos, data_system_controller_infos):
        self.config = config

        self.data_system_client = AsyncTransferQueueClient(
            client_id="RolloutManager",
            controller_infos=data_system_controller_infos[0],
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)

        self.async_rollout_workers = []
        num_workers = self.config.rollout_agent_num_workers
        for i in range(num_workers):
            self.async_rollout_workers.append(AsyncRolloutWorker.remote(config, data_system_controller_infos))

    def generate_sequences(self, data_meta):
        data_meta_chunkes = data_meta.chunk(len(self.async_rollout_workers))
        data_metas = ray.get(
            [
                worker.generate_sequences.remote(data_meta_chunk)
                for worker, data_meta_chunk in zip(self.async_rollout_workers, data_meta_chunkes, strict=True)
            ]
        )
        batch_meta = BatchMeta.concat(data_metas)
        logger.info(f"batch_meta: {batch_meta}")

        return batch_meta


class Trainer:
    def __init__(self, config):
        self.config = config
        self.data_system_client = self._initialize_data_system()
        self.actor_rollout_wg = ActorRolloutRefWorker()
        self.async_rollout_manager = RolloutManager(
            self.config,
            self.data_system_storage_unit_infos,
            self.data_system_controller_infos,
        )

    def _initialize_data_system(self):
        # TODO (TQStorage): provide a general data system initialization utility function
        # 1. 初始化TransferQueueStorage
        total_storage_size = self.config.global_batch_size * self.config.num_global_batch * self.config.num_n_samples
        self.data_system_storage_units = {}
        storage_placement_group = get_placement_group(self.config.num_data_storage_units, num_cpus_per_actor=1)
        for storage_unit_rank in range(self.config.num_data_storage_units):
            # TransferQueueStorage通过Ray拉起，是一个ray.remote修饰的类
            storage_node = SimpleStorageUnit.options(
                placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
            ).remote(storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units))
            self.data_system_storage_units[storage_unit_rank] = storage_node
            logger.info(f"TransferQueueStorageSimpleUnit #{storage_unit_rank} has been created.")

        # 2. 初始化TransferQueueController
        # 这里支持多controller实例以实现负载均衡，支持大规模扩展。不同controller可分配至不同RL计算任务
        self.data_system_controllers = {}
        controller_placement_group = get_placement_group(self.config.num_data_controllers, num_cpus_per_actor=1)
        for controller_rank in range(self.config.num_data_controllers):
            self.data_system_controllers[controller_rank] = TransferQueueController.options(
                placement_group=controller_placement_group, placement_group_bundle_index=controller_rank
            ).remote(
                global_batch_size=self.config.global_batch_size,
                num_global_batch=self.config.num_global_batch,
                num_n_samples=self.config.num_n_samples,
            )
            logger.info(f"TransferQueueController #{controller_rank} has been created.")

        # 3. 准备必要信息
        self.data_system_controller_infos = process_zmq_server_info(self.data_system_controllers)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})  # Note: Need to generate a new DictConfig
        # with allow_objects=True to maintain ZMQServerInfo instance. Otherwise it will be flattened to dict
        tq_config.controller_infos = self.data_system_controller_infos
        tq_config.storage_unit_infos = self.data_system_storage_unit_infos
        self.config = OmegaConf.merge(tq_config, self.config)

        # 4. 创建Client
        # TODO (TQStorage): Now it seems we cannot transmit the same client instance to multiple places,
        #                   otherwise ray will report serialization error. Need to check and fix it in the future.

        # TODO (TQStorage): TransferQueueClient needs a single controller_infos, while TransferQueueStorageManager
        #                   requires all controller_infos. Need to reconcile.
        self.data_system_client = AsyncTransferQueueClient(
            client_id="Trainer",
            controller_infos=self.data_system_controller_infos[0],
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)

        return self.data_system_client

    def fit(self):
        for epoch in range(1):
            train_dataloader = 1
            for step in range(train_dataloader):
                input_ids = (
                    torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [100, 111], [200, 222], [300, 333]])
                ) * (step + 1)
                input_ids_repeated = torch.repeat_interleave(input_ids, self.config.num_n_samples, dim=0)
                prompt_batch = TensorDict(
                    {"input_ids": input_ids_repeated, "attention_mask": input_ids_repeated},
                    batch_size=input_ids_repeated.size(0),
                )

                asyncio.run(self.data_system_client.async_put(data=prompt_batch, global_step=step))

                logger.info("demo put prompts ok! ")
                time.sleep(5)

                batch_meta = asyncio.run(
                    self.data_system_client.async_get_meta(
                        data_fields=["input_ids", "attention_mask"],
                        batch_size=self.config.global_batch_size * self.config.num_n_samples,
                        global_step=step,
                        get_n_samples=False,
                        task_name="generate_sequences",
                    )
                )
                logger.info(f"demo get meta {batch_meta}")

                # Simulate calling the generate sequences task of the worker group
                if not self.config.async_rollout_mode:
                    batch_meta = self.actor_rollout_wg.actor_rollout_wg_generate_sequences(
                        batch_meta, self.data_system_client
                    )
                else:
                    batch_meta = self.async_rollout_manager.generate_sequences(batch_meta)
                log_prob_meta = asyncio.run(
                    self.data_system_client.async_get_meta(
                        data_fields=["input_ids", "attention_mask", "generate_sequences_ids"],
                        batch_size=self.config.global_batch_size * self.config.num_n_samples,
                        global_step=step,
                        get_n_samples=False,
                        task_name="compute_old_log_prob",
                    )
                )
                logger.info(f"demo get log prob meta: {log_prob_meta}")

                # Simulate calling the compute old log prob task of the worker group
                old_log_prob_meta = self.actor_rollout_wg.actor_rollout_wg_compute_old_log_prob(
                    log_prob_meta, self.data_system_client
                )

                batch_meta = batch_meta.union(old_log_prob_meta)

                # 对于主控的client，通知所有controller进行数据状态清空，主控返回metadata；
                # client再根据metadata通知所有storage unit清空
                # client选择一个主controller拿到metadata，其他的controller直接清空不用返回metadata即可
                asyncio.run(self.data_system_client.async_clear(global_step=step))
                logger.info("clear ok! ")
        logger.info("demo done!")
        return batch_meta


if __name__ == "__main__":
    # NOTE: you may choose to set async_rollout_mode=True to test the async rollout mode that mimics
    # AgentLoopManager in verl

    config_str = """
      global_batch_size: 8
      num_global_batch: 1 
      num_data_storage_units: 2
      num_data_controllers: 1
      async_rollout_mode: True
      rollout_agent_num_workers: 2
      num_n_samples: 2

    """
    dict_conf = OmegaConf.create(config_str)

    trainer = Trainer(dict_conf)
    trainer.fit()

    ray.shutdown()
