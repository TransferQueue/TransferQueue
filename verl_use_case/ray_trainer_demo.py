from queue import Queue
from omegaconf import OmegaConf

from single_controller.ray import RayWorkerGroup, RayClassWithInitArgs, RayResourcePool
from single_controller.ray.base import create_colocated_worker_cls, create_colocated_worker_cls_fused
from workers.megatron_worker_demo import ActorRolloutRefWorker


class SentenceIterator:
    def __init__(self, train_batch_size):
        self.sentences = [
            "今天天气晴朗，适合户外活动。",
            "学习编程需要耐心和坚持。",
            "中国有悠久的历史和文化。",
            "健康饮食对身体健康非常重要。",
            "阅读是获取知识的最佳途径之一。",
            "大自然的美景总是让人心旷神怡。",
            "科技创新正在改变我们的生活方式。",
            "友谊是人生中最珍贵的财富之一。"
        ]
        self.start_index = 0
        self.batch_size = train_batch_size
        self.count = len(self.sentences)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_index < self.count:
            result = self.sentences[self.start_index:self.start_index+self.batch_size]
            self.start_index += self.batch_size
            return result
        else:
            raise StopIteration


class RayPPOTrainer:
    def __init__(
            self,
            config,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
        """

        self.hybrid_engine = True
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        self.config = config
        self.device_name = self.config.trainer.device
        self.total_training_steps = self.config.trainer.train_steps

        #################################################################################
        # Data System：Trainer类初始化时，基于Ray拉起Controller以及Storage；
        #################################################################################
        self._initialize_data_system()

        #################################################################################
        # Data System：提供iteration感知能力，记录当前actor训练的iteration
        #################################################################################
        self.iteration_record_queue = Queue(maxsize=self.config.algorithm.staleness_threshold)

        self.train_dataloader = SentenceIterator(self.config.data.train_batch_size)

    #################################################################################
    # Data System：初始化逻辑
    #################################################################################
    def _initialize_data_system(self):
        from utils.data_system import TransferQueueController, TransferQueueStorageSimpleUnit
        # 1. 初始化TransferQueueStorage
        total_storage_size = (self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n *
                              self.config.algorithm.staleness_threshold)
        self.data_system_storage_units = {}
        for storage_unit_rank in range(self.config.trainer.num_data_storage_units):
            # TransferQueueStorage通过Ray拉起，是一个ray.remote修饰的类
            storage_node = TransferQueueStorageSimpleUnit.remote(
                storage_unit_id=storage_unit_rank,
                storage_size=total_storage_size // self.config.trainer.num_data_storage_units + 1
            )
            self.data_system_storage_units[storage_unit_rank] = storage_node

        # 2. 初始化TransferQueueController
        # 这里支持多controller实例以实现负载均衡，支持大规模扩展。不同controller可分配至不同RL计算任务
        self.data_system_controllers = {}
        for controller_rank in range(self.config.trainer.num_data_controllers):
            self.data_system_controllers[controller_rank] = TransferQueueController.remote(
                controller_id=controller_rank,
                num_storage_units=self.config.trainer.num_data_storage_units,
                global_batch_size=self.config.data.train_batch_size,
                num_global_batch=self.config.algorithm.staleness_threshold,
                num_n_samples=self.config.actor_rollout_ref.rollout.n,
            )

        # 3. 将Controller注册至各个Storage
        # 每个Storage Unit拿到所有Controller的handler，通过Ray拿到对应的IP+端口，之后建立ZMQ Socket进行消息传输
        from utils.data_system import process_zmq_server_info

        self.data_system_controller_infos = process_zmq_server_info(self.data_system_controllers)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        # TODO: 待实现后打开注释
        # ray.get([storage_unit.register_controller_info.remote(self.data_system_controller_infos) for storage_unit in
        #          self.data_system_storage_units.values()])

        # 4. 创建Client
        from utils.data_system import TransferQueueClient
        self.data_system_client = TransferQueueClient(
            client_id='Trainer',
            controller_info=self.data_system_controller_infos[0],
            storage_info=self.data_system_storage_unit_infos,
            dp_world_size=None,
            num_dp_groups=None,
            dp_rank=None,
            rank_id=None
        )

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        ray_worker_group_count = 1
        actor_rollout_ref_cls = RayClassWithInitArgs(
            cls=ActorRolloutRefWorker,
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
            data_system_controller_infos=self.data_system_controller_infos[ray_worker_group_count % len(self.data_system_controller_infos)],
            data_system_storage_unit_infos=self.data_system_storage_unit_infos,
        )

        # resource_pool = RayResourcePool(process_on_nodes=[2])
        resource_pool = RayResourcePool(process_on_nodes=[self.config.trainer.num_gpus_per_node])
        # create colocated workers
        cls_dict = {"actor_rollout": actor_rollout_ref_cls}
        ray_cls_with_init = create_colocated_worker_cls_fused(cls_dict)

        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        wg_dict = RayWorkerGroup(resource_pool=resource_pool,
                                 ray_cls_with_init=ray_cls_with_init,
                                 device_name=self.device_name,
                                 data_system_controller_infos=self.data_system_controller_infos[ray_worker_group_count % len(self.data_system_controller_infos)],
                                 data_system_storage_unit_infos=self.data_system_storage_unit_infos,
                                 iteration_record_queue=self.iteration_record_queue,
                                 **wg_kwargs,
                                 )
        spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())

        self.actor_rollout_wg = spawn_wg["actor_rollout"]

        # self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False

    ######################################################################################################
    # Data System：简单展示新的主控逻辑（共部署场景）
    # 修改量较小，成本较低
    ######################################################################################################
    def fit_collocated_with_data_system(self):

        self.global_steps = 0

        # # add tqdm
        # progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            # TODO: 构造假的data_loader --已构造
            for prompt_batch in self.train_dataloader:
                print(f"prompt_batch: {prompt_batch}")
                # 问题：prompt_batch的大小是global_batch_size？跟experience_count的关系？是否能兼容verl worker_group dispatch和collect的逻辑？
                # 共卡场景下 prompt_batch的大小就是global_batch_size，experience_count=global_batch_size
                # TODO:构造一个假的storage_unit
                self.data_system_client.put_prompts(data=prompt_batch, global_step=self.global_steps,
                                                    n_samples_per_prompt=self.config.actor_rollout_ref.rollout.n)

                data_meta = self.data_system_client.get_meta(
                    data_columns=['prompt_token_ids', 'responses_token_ids', 'attention_mask', 'position_ids'],
                    experience_count=self.config.data.train_batch_size / self.config.actor_rollout_ref.rollout.dp_world_size,
                    # 每路DP的样本条数; 在共置场景为Global Batch Size / DP
                    dp_world_size=self.config.actor_rollout_ref.rollout.dp_world_size,  # DP总数
                    get_n_samples=False,
                    task_name='compute_log_prob',
                    schedule_policy='DP_balance'
                )

                print(f"trainer data_meta size: {data_meta.size} | global index is: {data_meta.global_indexes} "
                      f"| local index: {data_meta.local_indexes} | data cloumn: {data_meta.columns} "
                      f"| storage unit rank: {data_meta.storage_unit_rank}")
                ## trainer data_meta size: 4 | global index is: [0, 1, 2, 3] | local index: [0, 1, 2, 3] | data cloumn: ['prompt_token_ids', 'responses_token_ids', 'attention_mask', 'position_ids'] | storage unit rank: [0, 0, 0, 0] ##
                old_log_prob = self.actor_rollout_wg.compute_log_prob(data_meta)
                print(f"old_log_prob: {old_log_prob}")

                self.global_steps += 1
                ...