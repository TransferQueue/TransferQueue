from verl.single_controller.ray.base import RayWorkerGroup

class ActorWorkerGroup(RayWorkerGroup):
    def __init__(
        self,
        resource_pool: RayResourcePool,
        ray_cls_with_init: RayClassWithInitArgs,
        bin_pack: bool,
        name_prefix: str,
        detached,
        worker_names,
        worker_handles: list[ray.actor.ActorHandle],
        ray_wait_register_center_timeout: int,
        data_system_controller_infos,
        data_system_storage_unit_infos,
        **kwargs,
    ):
        super().__init__(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            bin_pack=bin_pack,
            name_prefix=name_prefix,
            detached=detached,
            worker_names=worker_names,
            worker_handles=worker_handles,
            ray_wait_register_center_timeout=ray_wait_register_center_timeout,
            **kwargs
        )


        ######################################################################################################
        # Data System：初始化数据系统Client
        ######################################################################################################
        self.data_system_controller_infos = data_system_controller_infos
        self.data_system_storage_unit_infos = data_system_storage_unit_infos
        self._build_data_system_client()
        pass

    def start(self):
        if self._is_actor:
            self.process_compute_log_prob_thread = Thread(target=self.compute_log_prob_separated,
                                                 name="compute_log_prob_thread",
                                                 daemon=True)
            self.process_compute_log_prob_thread.start()

            self.process_update_actor_thread = Thread(target=self.update_actor_separated,
                                                 name="update_actor_thread",
                                                 daemon=True)
            self.process_update_actor_thread.start()

        if self._is_rollout:
            self.process_generate_sequences_thread = Thread(target=self.generate_sequences_separated,
                                                 name="generate_sequences_thread",
                                                 daemon=True)
            self.process_generate_sequences_thread.start()

        if self._is_ref:
            self.process_compute_ref_log_prob_thread = Thread(target=self.compute_ref_log_prob_separated,
                                                 name="compute_ref_log_prob",
                                                 daemon=True)
            self.process_compute_ref_log_prob_thread.start()

    def compute_log_prob_separated(self):
        ######################################################################################################
        # Data System：在这里，提供一个子主控模块，基本完全仿照之前ray_trainer中的主控进行编写，只不过只维护自己的任务
        ######################################################################################################
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")


        while True:
            with marked_timer("step", timing_raw):
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    ##########################################################################################
                    # Data System：在主控中根据算法要求，为每一个任务取出对应的MetaData，并基于现有Dispatch功能分发
                    ##########################################################################################
                    log_prob_data_meta, current_global_step = self.data_system_client.get_meta(
                        data_columns=['prompt_token_ids', 'attention_mask', 'position_ids'],
                        experience_count=self.config.data.actor_compute_log_prob_dispatch_size,  # 每路DP的样本条数; 在分离场景为mbs,
                        dp_world_size=self.config.actor.rollout.dp_world_size,  # DP总数
                        num_dp_groups=None,  # 通过主控分发，无需声明DP域大小
                        dp_rank=None,  # 通过主控分发，无需指定自身的DP rank
                        rank_id=None,
                        get_n_samples=False,
                        schedule_policy='DP_balance'
                    )
                    self.actor_rollout_wg.compute_log_prob(log_prob_data_meta)

    def compute_log_prob(self):
        # 实现ActorRolloutRefWorker内原有compute_log_prob的功能
        # 是否将反注册过程颠倒过来，主动进行Dispatch修饰？
        pass


class RolloutWorkerGroup(RayWorkerGroup):
    pass


