import pytest
import sys
import zmq
import time
import torch
import tensordict
from threading import Thread
from tensordict import TensorDict
from pathlib import Path

# Import your classes here
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.data_system import (
    TransferQueueClient,
    ZMQServerInfo,
    ZMQRequestType,
    ZMQMessage,
    FieldMeta,
    SampleMeta,
    BatchMeta,
)



# Mock Controller for Client Unit Testing
class MockController:
    def __init__(self, controller_id="controller_0"):
        self.controller_id = controller_id
        self.context = zmq.Context()
        
        # Socket for data requests
        self.request_socket = self.context.socket(zmq.ROUTER)
        self.request_port = self._bind_to_random_port(self.request_socket)

        self.zmq_server_info = ZMQServerInfo.create(
            role="TransferQueueController",
            id=controller_id,
            ip="127.0.0.1",
            ports={
                "request_handle_socket": self.request_port,
            }
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
                        response_body = {"message": f"clear ok"}
                        response_type = ZMQRequestType.CLEAR_META_RESPONSE
                    
                    # Send response
                    response_msg = ZMQMessage.create(
                        request_type=response_type,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body=response_body
                    )
                    self.request_socket.send_multipart([identity, response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception:
                if self.running:
                    pass

    def _mock_batch_meta(self, request_body):
        batch_size = request_body.get('batch_size', 1)
        data_fields = request_body.get('data_fields', [])

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
                storage_id='storage_0',
                local_index=i,
                input_fields={field.name: field for field in fields}
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

        self.zmq_server_info = ZMQServerInfo.create(
            role="TransferQueueStorage",
            id=storage_id,
            ip="127.0.0.1",
            ports={
                "put_get_socket": self.data_port,
            }
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
                        body=response_body
                    )
                    self.data_socket.send_multipart([identity, response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception:
                if self.running:
                    pass

    def _handle_get_data(self, request_body):
        """Handle GET_DATA request by retrieving stored data"""
        local_indexes = request_body.get('local_indexes', [])
        fields = request_body.get('fields', [])
        
        # Retrieve the data (simulated with zeros)
        result_data = TensorDict({
            field: torch.zeros(len(local_indexes)) for field in fields
        })
        
        return {"message": {"data": result_data}}

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
    client_id = f"client_0"
    
    client = TransferQueueClient(
        client_id=client_id,
        controller_infos={mock_controller.controller_id: mock_controller.zmq_server_info},
        storage_infos={mock_storage.storage_id: mock_storage.zmq_server_info}
    )
    
    # Give some time for connections to establish
    time.sleep(0.5)
    
    yield client, mock_controller, mock_storage

# Test basic functionality
def test_client_initialization(client_setup):
    """Test client initialization and connection setup"""
    client, mock_controller, mock_storage = client_setup
    
    assert client.client_id is not None
    assert mock_controller.controller_id in client._controllers
    assert mock_storage.storage_id in client._storages

def test_put_and_get_data(client_setup):
    """Test basic put and get operations"""
    client, _, _ = client_setup
    
    # Create test data
    test_data = TensorDict({
        'tokens': torch.randint(0, 100, (5, 128)),
        'labels': torch.randint(0, 100, (5, 128))
    }, batch_size=5)
    
    # Test put operation
    client.put(
        data=test_data,
        data_fields=['tokens', 'labels'],
        global_step=0
    )
    
    # Get metadata for retrieving data
    metadata = client.get_meta(
        data_fields=['tokens', 'labels'],
        batch_size=5,
        global_step=0
    )
    
    # Test get operation
    result = client.get_data(metadata)
    
    # Verify result structure
    assert 'tokens' in result
    assert 'labels' in result
    assert result.batch_size[0] == 5

def test_get_meta(client_setup):
    """Test metadata retrieval"""
    client, _, _ = client_setup

    # Test get_meta operation
    metadata = client.get_meta(
        data_fields=['tokens', 'labels'],
        batch_size=10,
        global_step=0
    )

    # Verify metadata structure
    assert hasattr(metadata, 'storage_meta_groups')
    assert hasattr(metadata, 'global_indexes')
    assert hasattr(metadata, 'fields')
    assert hasattr(metadata, 'size')
    assert len(metadata.global_indexes) == 10

def test_clear_operation(client_setup):
    """Test clear operation"""
    client, _, _ = client_setup
    
    # Test clear operation
    client.clear(global_step=0)

# Test with multiple controllers and storage units
def test_multiple_servers():
    """Test client with multiple controllers and storage units"""
    # Create multiple mock servers
    controllers = [MockController(f"controller_{i}") for i in range(2)]
    storages = [MockStorage(f"storage_{i}") for i in range(3)]
    
    try:
        # Create client with multiple servers
        client_id = f"client_test_multiple_servers"
        
        controller_infos = {c.controller_id: c.zmq_server_info for c in controllers}
        storage_infos = {s.storage_id: s.zmq_server_info for s in storages}
        
        client = TransferQueueClient(
            client_id=client_id,
            controller_infos=controller_infos,
            storage_infos=storage_infos
        )
        
        # Give time for connections
        time.sleep(1.0)
        
        # Verify connections
        assert len(client._controllers) == 2
        assert len(client._storages) == 3
        
        # Test basic operation
        test_data = TensorDict({
            'tokens': torch.randint(0, 100, (5, 128))
        }, batch_size=5)
        
        # Test put operation
        client.put(
            data=test_data,
            data_fields=['tokens'],
            global_step=0
        )
        
    finally:
        # Clean up
        for c in controllers:
            c.stop()
        for s in storages:
            s.stop()

# Test error handling
def test_put_without_required_params(client_setup):
    """Test put operation without required parameters"""
    client, _, _ = client_setup
    
    # Create test data
    test_data = TensorDict({
        'tokens': torch.randint(0, 100, (5, 128))
    }, batch_size=5)
    
    # Test put without data_fields and global_step (should fail)
    with pytest.raises(AssertionError):
        client.put(data=test_data)

# Test tensor input
def test_put_tensor_input(client_setup):
    """Test put operation with tensor input (not TensorDict)"""
    client, _, _ = client_setup
    
    # Create test data as tensor
    test_data = torch.randint(0, 100, (5, 128))
    
    # Test put operation with tensor input
    client.put(
        data=test_data,
        data_fields=['tokens'],
        global_step=0
    )