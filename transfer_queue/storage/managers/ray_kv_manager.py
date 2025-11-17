import ray
from typing import Any
from tensordict import TensorDict
from transfer_queue.metadata import BatchMeta

from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory

@TransferQueueStorageManagerFactory.register("RAY")
class RayKVStorageManager(KVStorageManager):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        keys = self._generate_keys(metadata)
        values = self._generate_values(data)

        await self.storage_client.put.remote(keys=keys, values=values)

        per_field_dtypes = {}
        per_field_shapes = {}
        for global_idx in metadata.global_indexes:
            per_field_dtypes[global_idx] = {}
            per_field_shapes[global_idx] = {}

        for field in data.keys():
            for i, data_item in enumerate(data[field]):
                global_idx = metadata.global_indexes[i]
                per_field_dtypes[global_idx][field] = data_item.dtype if hasattr(data_item, "dtype") else None
                per_field_shapes[global_idx][field] = data_item.shape if hasattr(data_item, "shape") else None

        await self.notify_data_update(
            partition_id=metadata.samples[0].partition_id if metadata.samples else "unknown",
            fields=list(data.keys()),
            global_indexes=metadata.global_indexes,
            dtypes=per_field_dtypes,
            shapes=per_field_shapes,
        )

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        keys = self._generate_keys(metadata)
        shapes, dtypes = self._get_shape_type_list(metadata)

        values = await self.storage_client.get.remote(keys=keys, shapes=shapes, dtypes=dtypes)

        return self._merge_tensors_to_tensordict(metadata, values)

    async def clear_data(self, metadata: BatchMeta) -> None:
        keys = self._generate_keys(metadata)

        await self.storage_client.clear.remote(keys=keys)
