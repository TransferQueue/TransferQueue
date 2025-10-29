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

import dataclasses
import logging
import os
from dataclasses import dataclass
from operator import itemgetter
from threading import Thread
from typing import Optional
from uuid import uuid4

import ray
import torch
import zmq
from ray.util import get_node_ip_address
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import SampleMeta
from transfer_queue.utils.utils import TransferQueueRole
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo, create_zmq_socket, get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# ZMQ timeouts (in seconds) and retry configurations
TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))


class StorageUnitData:
    """Storage unit for managing 2D data structure (samples × fields).

    This class provides efficient storage and retrieval of data in a 2D matrix format
    where rows represent samples (indexed by local_index) and columns represent fields.
    Each field contains a list of data items indexed by their local position.

    Data Structure Example:
        ┌─────────────┬─────────────┬─────────────┬─────────┐
        │ local_index │ field_name1 │ field_name2 │  ...    │
        ├─────────────┼─────────────┼─────────────┼─────────┤
        │ 0           │ item1       │ item2       │  ...    │
        │ 1           │ item3       │ item4       │  ...    │
        │ 2           │ item5       │ item6       │  ...    │
        └─────────────┴─────────────┴─────────────┴─────────┘
    """

    def __init__(self, storage_size: int):
        # Dict containing field names and corresponding data in the field
        # Format: {"field_name": [data_at_index_0, data_at_index_1, ...]}
        self.field_data: dict[str, list] = {}

        # Maximum number of elements stored in storage unit
        self.storage_size = storage_size

    def get_data(self, fields: list[str], local_indexes: list[int]) -> TensorDict[str, list]:
        """
        Get data from storage unit according to given fields and local_indexes.

        Args:
            fields: Field names used for getting data.
            local_indexes: Local indexes used for getting data.

        Returns:
            TensorDict with field names as keys, corresponding data list as values.
        """
        result: dict[str, list] = {}

        for field in fields:
            # Validate field name
            if field not in self.field_data:
                raise ValueError(
                    f"StorageUnitData get_data operation receive invalid field: {field} beyond {self.field_data.keys()}"
                )

            if len(local_indexes) == 1:
                # The unsqueeze op make the shape from n to (1, n)
                gathered_item = self.field_data[field][local_indexes[0]]
                if not isinstance(gathered_item, torch.Tensor):
                    result[field] = NonTensorStack(gathered_item)
                else:
                    result[field] = gathered_item.unsqueeze(0)
            else:
                gathered_items = list(itemgetter(*local_indexes)(self.field_data[field]))

                if gathered_items:
                    all_tensors = all(isinstance(x, torch.Tensor) for x in gathered_items)
                    if all_tensors:
                        result[field] = torch.nested.as_nested_tensor(gathered_items)
                    else:
                        result[field] = NonTensorStack(*gathered_items)

        return TensorDict(result)

    def put_data(self, field_data: TensorDict[str, list], local_indexes: list[int]) -> None:
        """
        Put or update data into storage unit according to given field_data and local_indexes.

        Args:
            field_data: Dict with field names as keys, corresponding data in the field as values.
            local_indexes: Local indexes used for putting data.
        """
        extracted_data = dict(field_data)

        for f, values in extracted_data.items():
            if f not in self.field_data:
                self.field_data[f] = [None] * self.storage_size

            for i, idx in enumerate(local_indexes):
                if idx < 0 or idx >= self.storage_size:
                    raise ValueError(
                        f"StorageUnitData put_data operation receive invalid local_index: {idx} beyond "
                        f"storage_size: {self.storage_size}"
                    )

                self.field_data[f][idx] = values[i]

    def clear(self, local_indexes: list[int]) -> None:
        """
        Clear data at specified local_indexes by setting all related fields to None.

        Args:
            local_indexes: local_indexes to clear.
        """
        # Validate local_indexes
        for idx in local_indexes:
            if idx < 0 or idx >= self.storage_size:
                raise ValueError(
                    f"StorageUnitData clear operation receive invalid local_index: {idx} beyond "
                    f"storage_size: {self.storage_size}"
                )

        # Clear data at specified local_indexes
        for f in self.field_data:
            for idx in local_indexes:
                self.field_data[f][idx] = None


@ray.remote(num_cpus=1)
class SimpleStorageUnit:
    """A storage unit that provides distributed data storage functionality.

    This class represents a storage unit that can store data in a 2D structure
    (samples × data fields) and provides ZMQ-based communication for put/get/clear operations.

    Note: We use Ray decorator (@ray.remote) only for initialization purposes.
    We do NOT use Ray's .remote() call capabilities - the storage unit runs
    as a standalone process with its own ZMQ server socket.

    Attributes:
        storage_unit_id: Unique identifier for this storage unit.
        storage_unit_size: Maximum number of elements that can be stored.
        storage_data: Internal StorageUnitData instance for data management.
        zmq_server_info: ZMQ connection information for clients.
    """

    def __init__(self, storage_unit_size: int):
        """Initialize a SimpleStorageUnit with the specified size.

        Args:
            storage_unit_size: Maximum number of elements that can be stored in this storage unit.
        """
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4().hex[:8]}"
        self.storage_unit_size = storage_unit_size

        self.storage_data = StorageUnitData(self.storage_unit_size)

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id=str(self.storage_unit_id),
            ip=get_node_ip_address(),
            ports={"put_get_socket": get_free_port()},
        )
        self._init_zmq_socket()
        self._start_process_put_get()

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controller/clients:
        - put_get_socket:
            Handle put/get requests from clients.
        """
        self.zmq_context = zmq.Context()

        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER)
        self.put_get_socket.bind(self.zmq_server_info.to_addr("put_get_socket"))

    def _start_process_put_get(self) -> None:
        """Create a daemon thread and start put/get process."""
        self.process_put_get_thread = Thread(
            target=self._process_put_get, name=f"StorageUnitProcessPutGetThread-{self.zmq_server_info.id}", daemon=True
        )
        self.process_put_get_thread.start()

    def _process_put_get(self) -> None:
        """Process put_get_socket request."""
        poller = zmq.Poller()
        poller.register(self.put_get_socket, zmq.POLLIN)

        while True:
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT * 1000))

            if self.put_get_socket in socks:
                identity, serialized_msg = self.put_get_socket.recv_multipart()

                try:
                    request_msg = ZMQMessage.deserialize(serialized_msg)
                    operation = request_msg.request_type
                    logger.debug(f"[{self.zmq_server_info.id}]: receive operation: {operation}, message: {request_msg}")

                    if operation == ZMQRequestType.PUT_DATA:
                        response_msg = self._handle_put(request_msg)
                    elif operation == ZMQRequestType.GET_DATA:
                        response_msg = self._handle_get(request_msg)
                    elif operation == ZMQRequestType.CLEAR_DATA:
                        response_msg = self._handle_clear(request_msg)
                    else:
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.PUT_GET_OPERATION_ERROR,
                            sender_id=self.zmq_server_info.id,
                            body={
                                "message": f"Storage unit id #{self.zmq_server_info.id} "
                                f"receive invalid operation: {operation}."
                            },
                        )
                except Exception as e:
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.PUT_GET_ERROR,
                        sender_id=self.zmq_server_info.id,
                        body={
                            "message": f"Storage unit id #{self.zmq_server_info.id} occur error in processing "
                            f"put/get/clear request, detail error message: {str(e)}."
                        },
                    )

                self.put_get_socket.send_multipart([identity, response_msg.serialize()])

    def _handle_put(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle put request, add or update data into storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Put data success response ZMQMessage.
        """
        try:
            local_indexes = data_parts.body["local_indexes"]
            field_data = data_parts.body["data"]  # field_data should be in {field_name: [real data]} format.

            self.storage_data.put_data(field_data, local_indexes)

            # After put operation finish, send a message to the client
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE, sender_id=self.zmq_server_info.id, body={}
            )

            return response_msg
        except Exception as e:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to put data into storage unit id "
                    f"#{self.zmq_server_info.id}, detail error message: {str(e)}"
                },
            )

    def _handle_get(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle get request, return data from storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Get data success response ZMQMessage, containing target data.
        """
        try:
            fields = data_parts.body["fields"]
            local_indexes = data_parts.body["local_indexes"]

            result_data = self.storage_data.get_data(fields, local_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={
                    "data": result_data,
                },
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to get data from storage unit id #{self.zmq_server_info.id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def _handle_clear(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle clear request, clear data in storage unit according to given local_indexes.

        Args:
            data_parts: ZMQMessage from client, including target local_indexes.

        Returns:
            Clear data success response ZMQMessage.
        """
        try:
            local_indexes = data_parts.body["local_indexes"]

            self.storage_data.clear(local_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={"message": f"Clear data in storage unit id #{self.zmq_server_info.id} successfully."},
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to clear data in storage unit id #{self.zmq_server_info.id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get the ZMQ server information for this storage unit.

        Returns:
            ZMQServerInfo containing connection details for this storage unit.
        """
        return self.zmq_server_info


@dataclass
class StorageMetaGroup:
    """
    Represents a group of samples stored in the same storage unit.
    Used to organize samples by their storage_id for efficient client operations.
    """

    storage_id: str
    sample_metas: list[SampleMeta] = dataclasses.field(default_factory=list)
    local_indexes: list[int] = dataclasses.field(default_factory=list)

    def add_sample_meta(self, sample_meta: SampleMeta, local_index: int) -> None:
        """Add a SampleMeta object to this storage group"""
        self.sample_metas.append(sample_meta)
        self.local_indexes.append(local_index)

    def get_batch_indexes(self) -> list[int]:
        """Get all internal indexes from stored SampleMeta objects"""
        return [meta.batch_index for meta in self.sample_metas]

    def get_global_indexes(self) -> list[int]:
        """Get all global indexes from stored SampleMeta objects"""
        return [meta.global_index for meta in self.sample_metas]

    def get_local_indexes(self) -> list[int]:
        """Get all local indexes from stored SampleMeta objects"""
        return self.local_indexes

    def get_field_names(self) -> list[str]:
        """Get all unique field names from stored SampleMeta objects"""
        all_fields: set[str] = set()
        for meta in self.sample_metas:
            all_fields.update(meta.fields.keys())
        return list(all_fields)

    def get_transfer_data(self, field_names: Optional[list[str]] = None) -> dict[str, list | dict]:
        """Convert metadata to transfer dictionary format.

        Creates a transfer_dict structure containing indexing and field information
        but without the actual field data. The field_data placeholder will be
        populated by the _add_field_data() function.

        Args:
            field_names: Optional list of field names to include. If None, includes all fields.

        Returns:
            Transfer dictionary with metadata structure:
                {
                    "batch_indexes": [batch_idx1, batch_idx2, ...],
                    "global_indexes": [global_idx1, global_idx2, ...],
                    "local_indexes": [local_idx1, local_idx2, ...],
                    "fields": ["field1", "field2", ...],
                    "field_data": {}  # Placeholder - actual data added by _add_field_data()
                }

        Example:
            >>> group = StorageMetaGroup("storage1")
            >>> # Add multiple samples with different batch/global indexes and storage locations
            >>> group.add_sample_meta(SampleMeta(batch_index=0, global_index=10, fields={"img": ...}), 4)
            >>> group.add_sample_meta(SampleMeta(batch_index=1, global_index=11, fields={"img": ...}), 5)
            >>> group.add_sample_meta(SampleMeta(batch_index=2, global_index=12, fields={"img": ...}), 6)
            >>> transfer_dict = group.get_transfer_data(["img"])
            >>> transfer_dict["local_indexes"]   # [4, 5, 6] - storage locations
            >>> transfer_dict["batch_indexes"]   # [0, 1, 2] - original data locations
            >>> transfer_dict["global_indexes"]  # [10, 11, 12] - global identifiers
        """
        if field_names is None:
            field_names = self.get_field_names()
        return {
            "batch_indexes": self.get_batch_indexes(),
            "global_indexes": self.get_global_indexes(),
            "local_indexes": self.get_local_indexes(),
            "fields": field_names,
            "field_data": {},  # Placeholder for field data to be filled later
        }

    @property
    def size(self) -> int:
        """Number of samples in this storage meta group"""
        return len(self.sample_metas)

    @property
    def is_empty(self) -> bool:
        """Check if this storage meta group is empty"""
        return len(self.sample_metas) == 0

    def __len__(self) -> int:
        """Number of samples in this storage meta group"""
        return self.size

    def __bool__(self) -> bool:
        """Truthiness based on whether group has samples"""
        return not self.is_empty

    def __str__(self) -> str:
        return f"StorageMetaGroup(storage_id='{self.storage_id}', size={self.size})"
