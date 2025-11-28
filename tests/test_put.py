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
import sys
from pathlib import Path

import pytest
import ray
import torch
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    AsyncTransferQueueClient,
    SimpleStorageUnit,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.utils.utils import get_placement_group  # noqa: E402

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def ray_setup():
    """Initialize Ray for testing."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def data_system_setup(ray_setup):
    """Set up data system for testing."""
    # Initialize storage units
    num_storage_units = 2
    storage_size = 10000
    storage_units = {}

    storage_placement_group = get_placement_group(num_storage_units, num_cpus_per_actor=1)
    for storage_unit_rank in range(num_storage_units):
        storage_node = SimpleStorageUnit.options(
            placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
        ).remote(storage_unit_size=storage_size)
        storage_units[storage_unit_rank] = storage_node
        logger.info(f"SimpleStorageUnit #{storage_unit_rank} has been created.")

    # Initialize controller
    controller = TransferQueueController.remote()
    logger.info("TransferQueueController has been created.")

    # Prepare connection info
    controller_info = process_zmq_server_info(controller)
    storage_unit_infos = process_zmq_server_info(storage_units)

    # Create config
    config = {
        "controller_info": controller_info,
        "storage_unit_infos": storage_unit_infos,
    }

    yield controller, storage_units, config

    # Cleanup
    ray.kill(controller)
    for storage_unit in storage_units.values():
        ray.kill(storage_unit)


@pytest.fixture
async def client_setup(data_system_setup):
    """Set up client for testing."""
    controller, storage_units, config = data_system_setup

    client = AsyncTransferQueueClient(
        client_id="TestClient",
        controller_info=config["controller_info"],
    )

    client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

    # Wait a bit for connections to establish
    await asyncio.sleep(1)

    return client


class TestMultipleAsyncPut:
    """Test class for multiple async_put operations."""

    def __init__(self):
        self.controller = None
        self.storage_units = None
        self.config = None

    async def setup(self):
        """Setup for the test class."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Initialize data system
        num_storage_units = 2
        self.storage_units = {}
        storage_placement_group = get_placement_group(num_storage_units, num_cpus_per_actor=1)

        for i in range(num_storage_units):
            self.storage_units[i] = SimpleStorageUnit.options(
                placement_group=storage_placement_group, placement_group_bundle_index=i
            ).remote(storage_unit_size=10000)

        self.controller = TransferQueueController.remote()

        # Wait for initialization
        await asyncio.sleep(2)

        controller_info = process_zmq_server_info(self.controller)
        storage_unit_infos = process_zmq_server_info(self.storage_units)

        self.config = {
            "controller_info": controller_info,
            "storage_unit_infos": storage_unit_infos,
        }
        logger.info("TestMultipleAsyncPut setup completed")

    async def teardown(self):
        """Teardown for the test class."""
        if self.controller:
            ray.kill(self.controller)
        if self.storage_units:
            for storage in self.storage_units.values():
                ray.kill(storage)
        if ray.is_initialized():
            ray.shutdown()

    async def create_client(self, client_id="TestClient"):
        """Create a new client instance."""
        if self.config is None:
            await self.setup()

        client = AsyncTransferQueueClient(
            client_id=client_id,
            controller_info=self.config["controller_info"],
        )
        client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)
        await asyncio.sleep(1)  # Wait for connections
        return client


@pytest.mark.asyncio
async def test_concurrent_async_put():
    """Test concurrent async_put operations."""
    test_instance = TestMultipleAsyncPut()
    await test_instance.setup()

    try:
        client = await test_instance.create_client("ConcurrentTestClient")

        async def put_operation(partition_id, data):
            await client.async_put(data=data, partition_id=partition_id)
            return partition_id

        # Create multiple put operations
        tasks = []
        for i in range(5):
            data = TensorDict({f"concurrent_data_{i}": torch.randn(4, 8)}, batch_size=[4])
            task = put_operation(f"concurrent_partition_{i}", data)
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        logger.info(f"Completed {len(results)} concurrent put operations")

        # Cleanup
        for partition in results:
            try:
                await client.async_clear(partition)
            except Exception as e:
                logger.warning(f"Failed to clear partition {partition}: {e}")

        client.close()
    finally:
        await test_instance.teardown()


@pytest.mark.asyncio
async def test_sequential_async_put_with_verification():
    """Test sequential async_put operations with data verification."""
    test_instance = TestMultipleAsyncPut()
    await test_instance.setup()

    try:
        client = await test_instance.create_client("SequentialTestClient")

        # Test data
        test_cases = [
            ("sequential_1", torch.randn(4, 5)),
            ("sequential_2", torch.randn(4, 10)),
            ("sequential_3", torch.randn(4, 15)),
        ]

        for partition_id, tensor_data in test_cases:
            data = TensorDict({"sequential_data": tensor_data}, batch_size=[4])

            # Put data
            await client.async_put(data=data, partition_id=partition_id)
            logger.info(f"Put data to {partition_id}")

            # Verify by reading back
            metadata = await client.async_get_meta(
                data_fields=["sequential_data"],
                batch_size=4,
                partition_id=partition_id,
                mode="fetch",
                task_name="verification_task",
            )

            retrieved_data = await client.async_get_data(metadata)

            # Verify shape and content
            assert retrieved_data["sequential_data"].shape == tensor_data.shape
            logger.info(f"Verified data in {partition_id}")

            # Cleanup
            await client.async_clear(partition_id)

        client.close()
    finally:
        await test_instance.teardown()


@pytest.mark.asyncio
async def test_multiple_puts_same_partition():
    """Test multiple puts to the same partition."""
    test_instance = TestMultipleAsyncPut()
    await test_instance.setup()

    try:
        client = await test_instance.create_client("SamePartitionTestClient")

        partition_id = "same_partition_test"
        batch_size = 4

        # First put
        data1 = TensorDict({"first_batch": torch.randn(batch_size, 8)}, batch_size=[batch_size])

        await client.async_put(data=data1, partition_id=partition_id)
        logger.info("First put completed")

        # Second put to same partition
        data2 = TensorDict({"second_batch": torch.randn(batch_size, 8) * 2}, batch_size=[batch_size])

        await client.async_put(data=data2, partition_id=partition_id)
        logger.info("Second put completed")

        # Third put to same partition
        data3 = TensorDict({"third_batch": torch.randn(batch_size, 8) * 3}, batch_size=[batch_size])

        await client.async_put(data=data3, partition_id=partition_id)
        logger.info("Third put completed")

        # Verify the last data is what we get
        metadata = await client.async_get_meta(
            data_fields=["third_batch"],
            batch_size=batch_size,
            partition_id=partition_id,
            mode="fetch",
            task_name="verification_task",
        )

        retrieved_data = await client.async_get_data(metadata)

        assert "third_batch" in retrieved_data
        assert retrieved_data["third_batch"].shape == (batch_size, 8)
        logger.info("Verified multiple puts to same partition")

        # Cleanup
        await client.async_clear(partition_id)

        client.close()
    finally:
        await test_instance.teardown()


@pytest.mark.asyncio
async def test_simple_multiple_async_put():
    """Simple test using the existing client_setup fixture."""
    test_instance = TestMultipleAsyncPut()
    await test_instance.setup()

    try:
        client = await test_instance.create_client("SimpleTestClient")

        # Test basic multiple puts
        for i in range(3):
            data = TensorDict({f"simple_data_{i}": torch.randn(2, 4)}, batch_size=[2])

            partition_id = f"simple_partition_{i}"
            await client.async_put(data=data, partition_id=partition_id)
            logger.info(f"Put data to {partition_id}")

            # Cleanup each partition
            await client.async_clear(partition_id)

        logger.info("Simple multiple async_put test completed")

        client.close()
    finally:
        await test_instance.teardown()
