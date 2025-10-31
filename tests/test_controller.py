import logging
import sys
from pathlib import Path

import pytest
import ray
import torch
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transfer_queue import TransferQueueController
from transfer_queue.controller import TQ_INIT_FIELD_NUM
from transfer_queue.utils.utils import ProductionStatus


@pytest.fixture(scope="function")
def ray_setup():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "1", "RAY_DEDUP_LOGS": "0"}},
        log_to_driver=True,
    )
    yield
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray has been shut down completely after test")


class TestTransferQueueController:
    def test_controller_with_single_partition(ray_setup):
        gbs = 8
        num_n_samples = 4
        seq_len = 16

        tq_controller = TransferQueueController.remote()

        # Test get metadata in insert mode
        partition_id = "train_0"
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="insert",
                get_n_samples=True,
            )
        )

        assert metadata.global_indexes == list(range(gbs * num_n_samples))
        assert metadata.samples[0].partition_id == "train_0"
        assert sum(
            [int(sample.fields.get("prompt_ids").production_status) for sample in metadata.samples]
        ) == int(ProductionStatus.NOT_PRODUCED)
        assert sum(
            [int(sample.fields.get("attention_mask").production_status) for sample in metadata.samples]
        ) == int(ProductionStatus.NOT_PRODUCED)
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id))
        assert partition_index_range == set(range(gbs * num_n_samples))

        print("✓ Initial get metadata correct")

        # Test update production status
        original_prompts = torch.randn(gbs, seq_len)
        prompts_repeated = torch.repeat_interleave(original_prompts, num_n_samples, dim=0)
        original_attention_mask = torch.ones(gbs, seq_len)
        attention_mask_repeated = torch.repeat_interleave(original_attention_mask, num_n_samples, dim=0)
        input_batch = TensorDict(
            {"prompt_ids": prompts_repeated, "attention_mask": attention_mask_repeated},
            batch_size=prompts_repeated.size(0)
        )

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                sample_indices=metadata.global_indexes,
                field_names=metadata.field_names,
            )
        )
        assert success == True
        partition = ray.get(tq_controller.get_partition.remote(partition_id))
        assert partition.production_status is not None
        assert partition.production_status.size(0) == gbs * num_n_samples
        assert partition.production_status.size(1) == TQ_INIT_FIELD_NUM
        assert torch.equal(
            sum(partition.production_status[:, :len(data_fields)]), torch.Tensor([gbs*num_n_samples, gbs*num_n_samples])
        )
        assert torch.equal(
            sum(partition.production_status[:, len(data_fields):]),
            torch.zeros((1 * (TQ_INIT_FIELD_NUM - len(data_fields))))
        )

        print(f"✓ Updated production status for partition {partition_id}")

        # Test get metadate in fetch mode
        gen_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=["prompt_ids"],
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="fetch",
                task_name="generate_sequences",
                get_n_samples=False,
            )
        )
        assert gen_meta.global_indexes == list(range(gbs * num_n_samples))
        assert gen_meta.samples[0].partition_id == "train_0"
        assert gen_meta.field_names == ["prompt_ids"]
        partition = ray.get(tq_controller.get_partition.remote(partition_id))
        assert torch.equal(partition.consumption_status["generate_sequences"], torch.ones(gbs * num_n_samples))
        print("✓ Get metadata in fetch mode correct")

        # Test get clear meta
        clear_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=[],
                partition_id=partition_id,
                mode="insert",
            )
        )
        assert clear_meta.global_indexes == list(range(gbs * num_n_samples))
        assert [sample.fields for sample in clear_meta.samples] == [{}] * (gbs * num_n_samples)
        print("✓ Clear metadata correct")

        # Test clear
        ray.get(tq_controller.clear.remote(partition_id))
        partition = ray.get(tq_controller.get_partition.remote(partition_id))
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id))
        assert partition_index_range == set()
        assert torch.all(partition.production_status == 0)
        assert torch.all(partition.consumption_status["generate_sequences"] == 0)
        print("✓ Clear correct")
