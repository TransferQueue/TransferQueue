import os
import socket

import ray
from omegaconf import OmegaConf

from ray_trainer_demo import RayPPOTrainer


def main(config):

    ray.init()

    # Initialize the PPO trainer.
    trainer = RayPPOTrainer(
        config=config,
        )
    # Initialize the workers of the trainer.
    trainer.init_workers()
    # Start the training process.
    trainer.fit_collocated_with_data_system()


if __name__ == "__main__":
    config_str = """
        data:
          train_batch_size: 4
        actor_rollout_ref:
          actor:
            ppo_mini_batch_size: 4
          rollout:
            n: 8
            dp_world_size: 2
        algorithm:
          staleness_threshold: 1
        trainer:
          total_epochs: 1
          train_steps: 1
          num_gpus_per_node: 2
          num_data_storage_units: 2
          num_data_controllers: 2
          ray_wait_register_center_timeout: 300
          device: npu
        """
    dict_conf = OmegaConf.create(config_str)

    main(dict_conf)
