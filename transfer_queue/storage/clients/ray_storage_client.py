from typing import Any

import ray
import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory

@ray.remote(max_concurrency=8)
class RayObjectRefStorage:
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

@StorageClientFactory.register("RAY")
@ray.remote(num_gpus=1)
class RayStorageClient(TransferQueueStorageKVClient):

    def __init__(self):
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. Please call ray.init() before creating RayStorageClient."
            )

        self.use_gpu = torch.cuda.is_available()

        # initialize actor
        try:
            self.storage_actor = ray.get_actor("RayObjectRefStorage")
        except ValueError:
            self.storage_actor = RayObjectRefStorage.options(
                name="RayObjectRefStorage",
                lifetime="detached",
                get_if_exists=False
            ).remote()

    def set_use_gpu(self, use_gpu: bool):
        """Allow runtime toggle of use_gpu for performance testing."""
        self.use_gpu = use_gpu

    def put(self, keys: list[str], values: list[Any]):
        """
        Store tensors to remote storage.
        Args:
            keys (list): List of string keys
            values (list): List of torch.Tensor on GPU(CUDA) or CPU
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError(f"keys and values must be lists, but got {type(keys)} and {type(values)}")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        obj_refs = []
        for v in values:
            if isinstance(v, torch.Tensor) and v.is_cuda and self.use_gpu:
                # GPU Tensor → use NIXL
                ref = ray.put(v, _tensor_transport="nixl")
            else:
                # others ：CPU tensor、non-tensor → ray_obj_store
                ref = ray.put(v)
            obj_refs.append(ref)

        ray.get(self.storage_actor.put_gpu_obj_ref.remote(keys, obj_refs))

    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Any]:
        """
        Retrieve objects from remote storage.
        Args:
            keys (list): List of string keys to fetch.
            shapes (list, optional): Ignored. For compatibility with KVStorageManager.
            dtypes (list, optional): Ignored. For compatibility with KVStorageManager.
        Returns:
            list: List of retrieved objects
        """

        if not isinstance(keys, list):
            raise ValueError(f"keys must be a list, but got {type(keys)}")

        gpu_obj_refs = ray.get(self.storage_actor.get_gpu_obj_ref.remote(keys))
        # values = ray.get(gpu_obj_refs)
        values = []
        for key, gpu_obj_ref in zip(keys, gpu_obj_refs):
            try:
                if self.use_gpu:
                    # GPU tensor
                    value = ray.get(gpu_obj_ref, _tensor_transport="nixl")
                else:
                    # CPU
                    value = ray.get(gpu_obj_ref)
                values.append(value)
            except Exception:
                # GPU non-tensors fallback to use ray_obj_store
                try:
                    value = ray.get(gpu_obj_ref)
                    values.append(value)
                except Exception as e:
                    raise RuntimeError(f"Failed to retrieve value for key '{key}': {e}") from e

        return values

    def clear(self, keys: list[str]):
        """
        Delete entries from storage by keys.
        Args:
            keys (list): List of keys to delete
        """
        ray.get(self.storage_actor.clear_gpu_obj_ref.remote(keys))