import pytest  
import ray  
import torch  
import sys
from pathlib import Path
from tensordict import TensorDict  

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.storage.clients.ray_storage_client import RayStorageClient
  
@pytest.fixture(scope="session")  
def ray_setup():  
    ray.init(ignore_reinit_error=True)  
    yield  
    ray.shutdown()  
  
@pytest.fixture  
def ray_storage_client(ray_setup):  
    config = {"device_id": 0}  
    client = RayStorageClient(config)  
    yield client

def test_ray_storage_put_get(ray_storage_client):   
    keys = ["tensor_0", "tensor_1", "tensor_2"]  
    values = [  
        torch.randn(10, 20),  
        torch.randn(5, 15),  
        torch.randn(8, 12) 
    ]  
      
    # TEST PUT   
    ray_storage_client.put(keys, values)  
      
    # TEST GET  
    shapes = [v.shape for v in values]  
    dtypes = [v.dtype for v in values]  
    retrieved = ray_storage_client.get(keys, shapes=shapes, dtypes=dtypes)  
      
    for original, retrieved_tensor in zip(values, retrieved):  
        torch.testing.assert_close(original, retrieved_tensor)

# Multi-client concurrent testing
def test_multiple_clients_concurrent(ray_setup):    
    num_clients = 3    
    clients = [RayStorageClient({"device_id": 0}) for _ in range(num_clients)]    
        
    for i, client in enumerate(clients):    
        keys = [f"client_{i}_tensor_{j}" for j in range(3)]    
        values = [torch.randn(10, 10) * i for _ in range(3)] 
        client.put(keys, values)    
        
    for i, client in enumerate(clients):    
        keys = [f"client_{i}_tensor_{j}" for j in range(3)]    
        shapes = [(10, 10)] * 3    
        dtypes = [torch.float32] * 3    
        retrieved = client.get(keys, shapes=shapes, dtypes=dtypes)    
            
        for tensor in retrieved:    
            assert not tensor.is_cuda  
            assert tensor.shape == (10, 10)

