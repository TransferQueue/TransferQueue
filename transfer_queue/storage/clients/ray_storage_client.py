from typing import Any

import ray
import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory

@ray.remote
class RayGpuObjectRefStorage:
    def __init__(self):
        self.storage_dict = {}

    def put_gpu_obj_ref(self, keys: list[str], gpu_obj_refs: list[ray.ObjectRef]):
        for key, gpu_obj_ref in zip(keys, gpu_obj_refs):
            self.storage_dict[key] = gpu_obj_ref

    def get_gpu_obj_ref(self, keys: list[str]) -> list[ray.ObjectRef]:
        obj_refs = [self.storage_dict.get(key, None) for key in keys]
        return obj_refs

    def clear_gpu_obj_ref(self, keys: list[str]):
        for key in keys:
            if key in self.storage_dict:
                del self.storage_dict[key]

@StorageClientFactory.register("RDT")
class RayStorageClient(TransferQueueStorageKVClient):

    def __init__(self, config: dict[str, Any]):  
        if not ray.is_initialized():  
            ray.init()  
    
        self.use_gpu = torch.cuda.is_available()  
        
        if self.use_gpu:  
            gpu_ids = ray.get_gpu_ids()  
            if gpu_ids:  
                self.device_id = gpu_ids[0]  
            else:  
                self.device_id = config.get("device_id", 0)  
            torch.cuda.set_device(self.device_id)  
            self.device = f"cuda:{self.device_id}"  
        else:  
            self.device_id = None  
            self.device = "cpu"  

    def _create_empty_tensorlist(self, shapes, dtypes):
        """
        Create a list of empty NPU tensors with given shapes and dtypes.
        Args:
            shapes (list): List of tensor shapes (e.g., [(3,), (2, 4)])
            dtypes (list): List of torch dtypes (e.g., [torch.float32, torch.int64])
        Returns:
            list: List of uninitialized NPU tensors
        """
        if len(dtypes) != len(shapes):
            raise ValueError("Length of dtypes must equal length of shapes")

        tensors: list[Tensor] = []
        for dtype, shape in zip(dtypes, shapes, strict=False):
            tensor = torch.empty(shape, dtype=dtype).to(self.device)
            tensors.append(tensor)
        return tensors

    def put(self, keys: list[str], values: list[Tensor]):
        """
        Store tensors to remote storage.
        Args:
            keys (list): List of string keys
            values (list): List of torch.Tensor on NPU
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        for value in values:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(value)}")

        # TODO: NIXL can only be initialized in an environment with GPU, even if data is transferred on the cpu.
        if self.use_gpu:  
            obj_refs = [ray.put(v, _tensor_transport="nixl") for v in values]  
        else:  
            obj_refs = [ray.put(v) for v in values]  
        # obj_refs = [ray.put(v, _tensor_transport="nixl") for v in values]

        storage = RayGpuObjectRefStorage.options(
            name = "RayGpuObjectRefStorage",
            get_if_exists = True,
            lifetime = "detached"
        ).remote()

        storage.put_gpu_obj_ref.remote(keys, obj_refs)

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Tensor]:
        """
        Retrieve tensors from remote storage.
        Args:
            keys (list): List of keys to fetch
            shapes (list): Expected shapes of returned tensors
            dtypes (list): Expected dtypes of returned tensors
        Returns:
            list: List of retrieved NPU tensors
        """
        if len(dtypes) != len(shapes):
            raise ValueError("Length of dtypes must equal length of shapes")

        values: list[Tensor] = self._create_empty_tensorlist(shapes=shapes, dtypes=dtypes)
        storage = ray.get_actor("RayGpuObjectRefStorage")

        gpu_obj_refs = ray.get(storage.get_gpu_obj_ref.remote(keys))
        # values = ray.get(gpu_obj_refs)
        values = ray.get(gpu_obj_refs, _tensor_transport="nixl")
        return values

    def clear(self, keys: list[str]):
        """
        Delete entries from storage by keys.
        Args:
            keys (list): List of keys to delete
        """
        storage = ray.get_actor("RayGpuObjectRefStorage")
        ray.get(storage.clear_gpu_obj_ref.remote(keys))
