import os
import datetime
from omegaconf import DictConfig, ListConfig

import torch

from verl_use_case.single_controller.base import Worker
from verl_use_case.single_controller.base.decorator import Dispatch, register
from verl_use_case.utils.utils import get_nccl_backend, get_device_id


def compute_log_prob(data):
    return "hello" + data + "world"


def split_args(chunk_size, *args, **kwargs):
    rank = torch.distributed.get_rank()
    split_args = args.chunk(chunk_size)
    return split_args[rank], kwargs


def register_faker(dispatch_mode):
    def decorator(func):
        def wrapper(*args, **kwargs):
            dp_world_size = torch.distributed.get_world_size()
            new_args = split_args(dp_world_size, *args)
            return func(*new_args, **kwargs)
        return wrapper
    return decorator


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """
    ######################################################################################################
    # Data System：入参增加数据系统句柄
    # 理论上，这个Client应该是Base类的能力，不应该属于megatron_workers；
    # 但由于目前缺乏对底层训推引擎的封装隔离，沉入到最下层的Worker不太合适，TODO 因此这里还是先在最顶层实现，等待后续重构
    # verl.workers.megatron_workers.ActorRolloutRefWorker <- verl.single_controller.base.megatron.worker.MegatronWorker
    # <- verl.single_controller.base.worker.Worker
    #
    # 一个可能更加合理的抽象: Trainer -> XXWorkerGroup （Ray进程组）-> XXWorker （RL任务）-> BaseWorker -> Adapter -> Megatron
    # 这样可以把数据系统放在BaseWorker里
    ######################################################################################################
    def __init__(self, config: DictConfig, role: str, data_system_controller_infos, data_system_storage_unit_infos, **kwargs):
        Worker.__init__(self)
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            # 运行demo时需要注释
            # rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            # 运行demo时需要注释
            # get_torch_device().set_device(rank)

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]

        # # normalize config
        # if self._is_actor and self._is_rollout:
        #     self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
        #     self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        #     if self.config.actor.get("ppo_micro_batch_size", None):
        #         self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
        #         self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
        #         self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
        #         self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size


        ######################################################################################################
        # Data System：初始化数据系统Client
        ######################################################################################################
        self.data_system_controller_infos = data_system_controller_infos
        self.data_system_storage_unit_infos = data_system_storage_unit_infos
        self._build_data_system_client()

    def _build_data_system_client(self):
        from utils.data_system import TransferQueueClient
        self.data_system_client_handlers = TransferQueueClient(
            client_id='ActorRolloutRefWorker',
            controller_info=self.data_system_controller_infos,
            storage_info=self.data_system_storage_unit_infos,
            dp_world_size=None,
            num_dp_groups=None,
            dp_rank=None,
            rank_id=None
        )

    # TODO: 可以不用构造dispatch_fn，直接在worker group里显式的做chunk --在Dispatch.FAKER_MEGATRON_COMPUTE_PROTO中实现了data_meta的chunk demo
    # 或者构造一个自定义的register，实现dispatch_fn（collect_fn不需要实现，简化了主控的流程） --已构造
    @register(dispatch_mode=Dispatch.FAKER_MEGATRON_COMPUTE_PROTO)
    def compute_log_prob(self, data_meta):
        print(f"rank {torch.distributed.get_rank()} | data_meta is: {data_meta.global_indexes}| size is {data_meta.size} "
              f"| local index: {data_meta.local_indexes} | data cloumn: {data_meta.columns} "
              f"| storage unit rank: {data_meta.storage_unit_rank}")
        ## rank 0 | data_meta is: [0, 1]| size is 2 | local index: [0, 1] | data cloumn: ['prompt_token_ids', 'responses_token_ids', 'attention_mask', 'position_ids'] | storage unit rank: [0, 0] ##
        ## rank 1 | data_meta is: [2, 3]| size is 2 | local index: [2, 3] | data cloumn: ['prompt_token_ids', 'responses_token_ids', 'attention_mask', 'position_ids'] | storage unit rank: [0, 0] ##
        ######################################################################################################
        # Data System：统一共置+分离，分离架构可以通过引入一个子主控解决调试问题
        ######################################################################################################

        data = self.data_system_client_handlers.get_data(data_meta)

        assert self._is_actor

        # we should always recompute old_log_probs when it is HybridEngine
        data = data.to(get_device_id())
        # TODO: 假的compute_log_prob函数 --已构造
        output = compute_log_prob(data=data)
        # output = DataProto.from_dict(
        #     tensors={"old_log_probs": output, "entropys": entropys},
        #     meta_info={"temperature": self.config.rollout.temperature},
        # )
        output = output.to("cpu")

        self.data_system_client_handlers.put(data=output, metadata=data_meta)

        get_torch_device().empty_cache()
