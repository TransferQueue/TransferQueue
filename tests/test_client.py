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

import sys
import time
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
import torch
import zmq
from tensordict import NonTensorStack, TensorDict

# Import your classes here
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import TransferQueueClient  # noqa: E402
from transfer_queue.metadata import (  # noqa: E402
    BatchMeta,
    FieldMeta,
    SampleMeta,
)
from transfer_queue.utils.zmq_utils import (  # noqa: E402
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
)

TEST_DATA = TensorDict(
    {
        "log_probs": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])],
        "variable_length_sequences": torch.nested.as_nested_tensor(
            [
                torch.tensor([-0.5, -1.2, -0.8]),
                torch.tensor([-0.3, -1.5, -2.1, -0.9]),
                torch.tensor([-1.1, -0.7]),
            ]
        ),
        "prompt_text": ["Hello world!", "This is a longer sentence for testing", "Test case"],
    },
    batch_size=[3],
)


# Mock Controller for Client Unit Testing
class MockController:
    def __init__(self, controller_id="controller_0"):
        self.controller_id = controller_id
        self.context = zmq.Context()

        # Socket for data requests
        self.request_socket = self.context.socket(zmq.ROUTER)
        self.request_port = self._bind_to_random_port(self.request_socket)

        self.zmq_server_info = ZMQServerInfo(
            role="TransferQueueController",
            id=controller_id,
            ip="127.0.0.1",
            ports={
                "request_handle_socket": self.request_port,
            },
        )

        self.running = True
        self.request_thread = Thread(target=self._handle_requests, daemon=True)
        self.request_thread.start()

    def _bind_to_random_port(self, socket):
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        return port

    def _handle_requests(self):
        poller = zmq.Poller()
        poller.register(self.request_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.request_socket in socks:
                    identity, serialized_msg = self.request_socket.recv_multipart()
                    request_msg = ZMQMessage.deserialize(serialized_msg)

                    # Determine response based on request type
                    if request_msg.request_type == ZMQRequestType.GET_META:
                        response_body = self._mock_batch_meta(request_msg.body)
                        response_type = ZMQRequestType.GET_META_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.GET_CLEAR_META:
                        response_body = self._mock_batch_meta(request_msg.body)
                        response_type = ZMQRequestType.GET_CLEAR_META_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.CLEAR_META:
                        response_body = {"message": "clear ok"}
                        response_type = ZMQRequestType.CLEAR_META_RESPONSE

                    # Send response
                    response_msg = ZMQMessage.create(
                        request_type=response_type,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body=response_body,
                    )
                    self.request_socket.send_multipart([identity, response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception as e:
                print(f"MockController ERROR: {e}")
                raise

    def _mock_batch_meta(self, request_body):
        batch_size = request_body.get("batch_size", 1)
        data_fields = request_body.get("data_fields", [])

        samples = []
        for i in range(batch_size):
            fields = []
            for field_name in data_fields:
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=None,
                    shape=None,
                    production_status=0,
                )
                fields.append(field_meta)
            sample = SampleMeta(
                global_step=0,
                global_index=i,
                fields={field.name: field for field in fields},
            )
            samples.append(sample)
        metadata = BatchMeta(samples=samples)

        return {"metadata": metadata}

    def stop(self):
        self.running = False
        time.sleep(0.2)  # Give thread time to stop
        self.request_socket.close()
        self.context.term()


# Mock Storage for Client Unit Testing
class MockStorage:
    def __init__(self, storage_id="storage_0"):
        self.storage_id = storage_id
        self.context = zmq.Context()

        # Socket for data operations
        self.data_socket = self.context.socket(zmq.ROUTER)
        self.data_port = self._bind_to_random_port(self.data_socket)

        self.zmq_server_info = ZMQServerInfo(
            role="TransferQueueStorage",
            id=storage_id,
            ip="127.0.0.1",
            ports={
                "put_get_socket": self.data_port,
            },
        )

        self.running = True
        self.data_thread = Thread(target=self._handle_data_requests, daemon=True)
        self.data_thread.start()

    def _bind_to_random_port(self, socket):
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        return port

    def _handle_data_requests(self):
        poller = zmq.Poller()
        poller.register(self.data_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.data_socket in socks:
                    identity, msg_bytes = self.data_socket.recv_multipart()
                    msg = ZMQMessage.deserialize(msg_bytes)

                    # Handle different request types
                    if msg.request_type == ZMQRequestType.PUT_DATA:
                        response_body = {"message": "Data stored successfully"}
                        response_type = ZMQRequestType.PUT_DATA_RESPONSE
                    elif msg.request_type == ZMQRequestType.GET_DATA:
                        response_body = self._handle_get_data(msg.body)
                        response_type = ZMQRequestType.GET_DATA_RESPONSE
                    elif msg.request_type == ZMQRequestType.CLEAR_DATA:
                        response_body = {"message": "Data cleared successfully"}
                        response_type = ZMQRequestType.CLEAR_DATA_RESPONSE

                    # Send response
                    response_msg = ZMQMessage.create(
                        request_type=response_type,
                        sender_id=self.storage_id,
                        receiver_id=msg.sender_id,
                        body=response_body,
                    )
                    self.data_socket.send_multipart([identity, response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"MockStorage running exception: {e}")
                else:
                    print(f"MockStorage ERROR: {e}")
                    raise

    def _handle_get_data(self, request_body):
        """Handle GET_DATA request by retrieving stored data"""
        local_indexes = request_body.get("local_indexes", [])
        fields = request_body.get("fields", [])

        result: dict[str, list] = {}
        for field in fields:
            gathered_items = [TEST_DATA[field][i] for i in local_indexes]

            if gathered_items:
                all_tensors = all(isinstance(x, torch.Tensor) for x in gathered_items)
                if all_tensors:
                    result[field] = torch.nested.as_nested_tensor(gathered_items)
                else:
                    result[field] = NonTensorStack(*gathered_items)

        return {"data": TensorDict(result)}

    def stop(self):
        self.running = False
        time.sleep(0.2)  # Give thread time to stop
        self.data_socket.close()
        self.context.term()


# Test Fixtures
@pytest.fixture
def mock_controller():
    controller = MockController()
    yield controller
    controller.stop()


@pytest.fixture
def mock_storage():
    storage = MockStorage()
    yield storage
    storage.stop()


@pytest.fixture
def client_setup(mock_controller, mock_storage):
    # Create client with mock controller and storage
    client_id = "client_0"

    client = TransferQueueClient(
        client_id=client_id,
        controller_info=mock_controller.zmq_server_info,
    )

    # Mock the storage manager to avoid handshake issues but mock all data operations
    with patch("transfer_queue.storage.AsyncSimpleStorageManager._connect_to_controllers"):
        config = {
            "controller_info": mock_controller.zmq_server_info,
            "storage_unit_infos": {mock_storage.storage_id: mock_storage.zmq_server_info},
        }
        client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

        # Mock all storage manager methods to avoid real ZMQ operations
        async def mock_put_data(data, metadata):
            pass  # Just pretend to store the data

        async def mock_get_data(metadata):
            # Return the test data when requested
            return TEST_DATA

        async def mock_clear_data(metadata):
            pass  # Just pretend to clear the data

        client.storage_manager.put_data = mock_put_data
        client.storage_manager.get_data = mock_get_data
        client.storage_manager.clear_data = mock_clear_data

    yield client, mock_controller, mock_storage


# Test basic functionality
def test_client_initialization(client_setup):
    """Test client initialization and connection setup"""
    client, mock_controller, mock_storage = client_setup

    assert client.client_id is not None
    assert client._controller is not None
    assert client._controller.id == mock_controller.controller_id


def test_put_and_get_data(client_setup):
    """Test basic put and get operations"""
    client, _, _ = client_setup

    # Test put operation
    client.put(data=TEST_DATA, global_step=0)

    # Get metadata for retrieving data
    metadata = client.get_meta(
        data_fields=["log_probs", "variable_length_sequences", "prompt_text"], batch_size=2, global_step=0
    )

    # Test get operation
    result = client.get_data(metadata)

    # Verify result structure
    assert "log_probs" in result
    assert "variable_length_sequences" in result
    assert "prompt_text" in result

    torch.testing.assert_close(result["log_probs"][0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(result["log_probs"][1], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(result["variable_length_sequences"][0], torch.tensor([-0.5, -1.2, -0.8]))
    torch.testing.assert_close(result["variable_length_sequences"][1], torch.tensor([-0.3, -1.5, -2.1, -0.9]))
    assert result["prompt_text"][0] == "Hello world!"
    assert result["prompt_text"][1] == "This is a longer sentence for testing"


def test_get_meta(client_setup):
    """Test metadata retrieval"""
    client, _, _ = client_setup

    # Test get_meta operation
    metadata = client.get_meta(data_fields=["tokens", "labels"], batch_size=10, global_step=0)

    # Verify metadata structure
    assert hasattr(metadata, "global_indexes")
    assert hasattr(metadata, "field_names")
    assert hasattr(metadata, "size")
    assert len(metadata.global_indexes) == 10


def test_clear_operation(client_setup):
    """Test clear operation"""
    client, _, _ = client_setup

    # Test clear operation
    client.clear(global_step=0)


# Test with single controller and multiple storage units
def test_single_controller_multiple_storages():
    """Test client with single controller and multiple storage units"""
    # Create single controller and multiple storage units
    controller = MockController("controller_0")
    storages = [MockStorage(f"storage_{i}") for i in range(3)]

    try:
        # Create client with single controller
        client_id = "client_test_single_controller"

        client = TransferQueueClient(client_id=client_id, controller_info=controller.zmq_server_info)

        # Mock the storage manager to avoid handshake issues but mock all data operations
        with patch("transfer_queue.storage.AsyncSimpleStorageManager._connect_to_controllers"):
            config = {
                "controller_info": controller.zmq_server_info,
                "storage_unit_infos": {s.storage_id: s.zmq_server_info for s in storages},
            }
            client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

            # Mock all storage manager methods to avoid real ZMQ operations
            async def mock_put_data(data, metadata):
                pass  # Just pretend to store the data

            async def mock_get_data(metadata):
                # Return some test data when requested
                return TensorDict({"tokens": torch.randint(0, 100, (5, 128))}, batch_size=5)

            async def mock_clear_data(metadata):
                pass  # Just pretend to clear the data

            client.storage_manager.put_data = mock_put_data
            client.storage_manager.get_data = mock_get_data
            client.storage_manager.clear_data = mock_clear_data

        # Verify controller is set
        assert client._controller is not None
        assert client._controller.id == controller.controller_id

        # Test basic operation
        test_data = TensorDict({"tokens": torch.randint(0, 100, (5, 128))}, batch_size=5)

        # Test put operation
        client.put(data=test_data, global_step=0)

    finally:
        # Clean up
        controller.stop()
        for s in storages:
            s.stop()


# Test error handling
def test_put_without_required_params(client_setup):
    """Test put operation without required parameters"""
    client, _, _ = client_setup

    # Create test data
    test_data = TensorDict({"tokens": torch.randint(0, 100, (5, 128))}, batch_size=5)

    # Test put without global_step (should fail)
    with pytest.raises(AssertionError):
        client.put(data=test_data)
