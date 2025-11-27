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
import numpy as np
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
        # 1. Pull real data from the storage plane through client based on data_meta
        data = asyncio.run(data_system_client.async_get_data(data_meta))
        logger.info(f"demo get data->generate_sequences {data}")

        output = generate_sequences(data["input_ids"])

        output = TensorDict(
            {
                "generate_sequences_ids": output,
                "non_tensor_data": torch.stack([NonTensorData("test_str") for _ in range(output.size(0))]),
                "nested_tensor": torch.nested.as_nested_tensor(
                    [torch.randn(1, 2) for _ in range(output.size(0))], layout=torch.jagged
                ),
            },
            batch_size=output.size(0),
        )

        # 2. Write results back to the storage plane based on data_meta
        asyncio.run(data_system_client.async_put(data=output, metadata=data_meta))
        data_meta.add_fields(output)
        logger.info("demo put data to storages done")

        return data_meta

    def actor_rollout_wg_compute_old_log_prob(self, data_meta, data_system_client):
        # 1. Pull real data from the storage plane through client based on data_meta
        data = asyncio.run(data_system_client.async_get_data(data_meta))
        logger.info(f"demo get data->old_log_prob {data}")

        output = compute_old_log_prob(data["input_ids"], data["generate_sequences_ids"])

        output = TensorDict({"old_log_prob": output}, batch_size=output.size(0))

        # 2. Write results back to the storage plane based on data_meta
        asyncio.run(data_system_client.async_put(data=output, metadata=data_meta))
        data_meta.add_fields(output)
        logger.info("demo put data to storages done")

        return data_meta


@ray.remote
class AsyncvLLMServer:
    def __init__(self, config, data_system_controller_info):
        self.config = config
        self.data_system_client = AsyncTransferQueueClient(
            client_id="AsyncvLLMServer",
            controller_info=data_system_controller_info,
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
                "nested_tensor": torch.nested.as_nested_tensor(
                    [torch.randn(1, 2) for _ in range(data.size(0))], layout=torch.jagged
                ),
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
        data_system_controller_info,
    ):
        self.async_vllm_server = AsyncvLLMServer.remote(
            config,
            data_system_controller_info,
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
    def __init__(self, config, data_system_storage_unit_infos, data_system_controller_info):
        self.config = config

        self.data_system_client = AsyncTransferQueueClient(
            client_id="RolloutManager",
            controller_info=data_system_controller_info,
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)

        self.async_rollout_workers = []
        num_workers = self.config.rollout_agent_num_workers
        for i in range(num_workers):
            self.async_rollout_workers.append(AsyncRolloutWorker.remote(config, data_system_controller_info))

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
            self.data_system_controller_info,
        )
    @classmethod
    def dict_to_tensordict(cls, data: dict[str, torch.Tensor | np.ndarray]) -> TensorDict:
        """
        Create a TensorDict from a dict of tensors and non_tensors.
        Note that this requires tensordict version at least 0.10
        """

        tensors_batch = {}
        batch_size = None
        from tensordict import NonTensorData, NonTensorStack
        for key, val in data.items():
            print(key, val)
            if isinstance(val, torch.Tensor | np.ndarray | NonTensorStack | list):
                tensors_batch[key] = val
            else:
                raise ValueError(f"Unsupported type in ata {type(val)}")

            if batch_size is None:
                batch_size = len(val)
            else:
                assert len(val) == batch_size

        if batch_size is None:
            batch_size = []
        else:
            batch_size = [batch_size]

        return TensorDict(tensors_batch, batch_size=batch_size)

    def _initialize_data_system(self):
        # 1. Initialize TransferQueueStorage
        total_storage_size = self.config.global_batch_size * self.config.num_global_batch * self.config.num_n_samples
        self.data_system_storage_units = {}
        storage_placement_group = get_placement_group(self.config.num_data_storage_units, num_cpus_per_actor=1)
        for storage_unit_rank in range(self.config.num_data_storage_units):
            storage_node = SimpleStorageUnit.options(
                placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
            ).remote(storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units))
            self.data_system_storage_units[storage_unit_rank] = storage_node
            logger.info(f"SimpleStorageUnit #{storage_unit_rank} has been created.")

        # 2. Initialize TransferQueueController (single controller only)

        # Sampler usage instructions:
        # For GRPO grouped sampling, you can initialize the controller with GRPOGroupNSampler:
        # Option 1: Pass sampler class (will be instantiated automatically)
        # self.data_system_controller = TransferQueueController.remote(sampler=GRPOGroupNSampler)

        # Option 2: Pass sampler instance (if you need custom configuration)
        # grpo_sampler = GRPOGroupNSampler()
        # self.data_system_controller = TransferQueueController.remote(sampler=grpo_sampler)

        # Then use sampling_config in get_meta calls:
        # sampling_config={"n_samples_per_prompt": 4}
        self.data_system_controller = TransferQueueController.remote()
        logger.info("TransferQueueController has been created.")

        # 3. Prepare necessary information
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})  # Note: Need to generate a new DictConfig
        # with allow_objects=True to maintain ZMQServerInfo instance. Otherwise it will be flattened to dict
        tq_config.controller_info = self.data_system_controller_info
        tq_config.storage_unit_infos = self.data_system_storage_unit_infos
        self.config = OmegaConf.merge(tq_config, self.config)

        # 4. Create client
        self.data_system_client = AsyncTransferQueueClient(
            client_id="Trainer",
            controller_info=self.data_system_controller_info,
        )

        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)
        # Note: The client contains ZMQ objects. Currently, we cannot transmit the same client instance
        # to multiple places, as this will cause serialization errors in Ray.
        # Workaround: If you need to use a client in multiple Ray actors or processes, create a separate
        # AsyncTransferQueueClient instance for each actor/process instead of sharing or transmitting the same instance.
        return self.data_system_client

    def fit(self):
        for epoch in range(1):
            train_dataloader = 1
            for step in range(train_dataloader):
                # Handle multi-modal data by storing them separately in data system,
                # and only keep the metadata in the main batch in "multi_modal_data".

                # Data Format for Deepeyes: {'multi_modal_data':array([{'image':[<PIL>,<PIL>]}, {'image':[<PIL>]}])}
                # It's better to transform PIL into tensor in DataLoader, so in the future it may become
                # {'multi_modal_data':array([{'image':[torch.Tensor, torch.Tensor]}, {'image':[torch.Tensor]}])}

                # 1. Split multi_modal_data into single items and put them into different partition
                import numpy as np
                batch_dict = {
                    "multi_modal_data": np.array(
                        [
                            {"image": [torch.randn(3, 4), torch.randn(3, 4)], "video": [torch.randn(3, 4, 5)]},
                            {"image": [torch.randn(4, 5)], "video": []},
                        ]
                    ),
                    "tensor": [torch.randn(4, 5), torch.randn(4, 5)]
                }

                multi_modal_batch_meta = []
                for mm_sample in batch_dict["multi_modal_data"]:
                    mm_keys = list(mm_sample.keys())
                    mm_sample_batch_meta = {}
                    for modality in mm_keys:
                        modality_data = mm_sample[modality]
                        if len(modality_data) > 0:
                            modality_partition_id = f"train_mm_{step - 1}_{modality}"
                            modality_tensordict = TensorDict({modality: modality_data}, batch_size=len(modality_data))

                            batch_meta = asyncio.run(
                                self.data_system_client.async_put(data=modality_tensordict, partition_id=modality_partition_id)
                            )
                            mm_sample_batch_meta[modality] = batch_meta

                            print(f"batch meta = {batch_meta}")
                    multi_modal_batch_meta.append(mm_sample_batch_meta)

                # replacing original multi-modal data
                from tensordict import NonTensorStack
                batch_dict["multi_modal_data"] = multi_modal_batch_meta

                print(f"batch meta into NonTensorStack= {batch_dict['multi_modal_data']}")


                batch: TensorDict = self.dict_to_tensordict(batch_dict)
                batch_meta = asyncio.run(self.data_system_client.async_put(data=batch, partition_id=f"train_{step - 1}"))

                data_0 = asyncio.run(self.data_system_client.async_get_data(batch_meta[0]))

                print(f"data_0 = {data_0}")
                print(f"data_0_mm_batch meta = {data_0['multi_modal_data']}")
                data_0_mm = asyncio.run(self.data_system_client.async_get_data(data_0['multi_modal_data'][0]['image']))   # 如果是tensor，tensordict会合到一起

                print(f"retrieved multi_modal data {data_0_mm['image']}")

                time.sleep(500)


                logger.info("demo put prompts ok! ")
                time.sleep(5)

                batch_meta = asyncio.run(
                    self.data_system_client.async_get_meta(
                        data_fields=["input_ids", "attention_mask"],
                        batch_size=self.config.global_batch_size * self.config.num_n_samples,
                        partition_id=f"train_{step}",
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
                        partition_id=f"train_{step}",
                        task_name="compute_old_log_prob",
                    )
                )
                logger.info(f"demo get log prob meta: {log_prob_meta}")

                # Simulate calling the compute old log prob task of the worker group
                old_log_prob_meta = self.actor_rollout_wg.actor_rollout_wg_compute_old_log_prob(
                    log_prob_meta, self.data_system_client
                )

                batch_meta = batch_meta.union(old_log_prob_meta)

                # Client notifies controller to clear data status, controller returns metadata;
                # Client then notifies the storage plane to clear based on metadata
                asyncio.run(self.data_system_client.async_clear(partition_id=f"train_{step}"))
                logger.info("clear ok! ")
        logger.info("demo done!")

        # Cleanup resources
        self.data_system_client.close()
        return batch_meta


if __name__ == "__main__":
    # NOTE: you may choose to set async_rollout_mode=True to test the async rollout mode that mimics
    # AgentLoopManager in verl
    config_str = """
      global_batch_size: 8
      num_global_batch: 1
      num_data_storage_units: 2
      async_rollout_mode: True
      rollout_agent_num_workers: 2
      num_n_samples: 2

    """
    dict_conf = OmegaConf.create(config_str)

    trainer = Trainer(dict_conf)
    trainer.fit()

    ray.shutdown()
