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
import concurrent.futures
import dataclasses
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from operator import itemgetter
from threading import Thread
from typing import Any, Callable, Optional
from uuid import uuid4

import ray
import torch
import zmq
from ray.util import get_node_ip_address
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta, SampleMeta
from transfer_queue.utils.utils import TransferQueueRole, limit_pytorch_auto_parallel_threads
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# ZMQ timeouts (in seconds) and retry configurations
TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))
TQ_STORAGE_HANDSHAKE_TIMEOUT = int(os.environ.get("TQ_STORAGE_HANDSHAKE_TIMEOUT", 30))
TQ_STORAGE_HANDSHAKE_RETRY_INTERVAL = int(os.environ.get("TQ_STORAGE_HANDSHAKE_RETRY_INTERVAL", 1))
TQ_STORAGE_HANDSHAKE_MAX_RETRIES = int(os.environ.get("TQ_STORAGE_HANDSHAKE_MAX_RETRIES", 3))
TQ_DATA_UPDATE_RESPONSE_TIMEOUT = int(os.environ.get("TQ_DATA_UPDATE_RESPONSE_TIMEOUT", 30))


class TransferQueueStorageManager(ABC):
    """Base class for storage layer. It defines the interface for data operations and
    generally provides handshake & notification capabilities."""

    def __init__(self, config: dict[str, Any]):
        self.storage_manager_id = f"TQ_STORAGE_{uuid4().hex[:8]}"
        self.config = config
        self.controller_info = config.get("controller_info", None)  # type: ZMQServerInfo

        self.data_status_update_socket = None
        self.controller_handshake_socket = None

        self.zmq_context = None
        self._connect_to_controller()

    def _connect_to_controller(self) -> None:
        """Initialize ZMQ sockets between storage unit and controller for handshake."""
        if not isinstance(self.controller_info, ZMQServerInfo):
            raise ValueError(f"controller_info should be ZMQServerInfo, but got {type(self.controller_info)}")

        try:
            # create zmq context
            self.zmq_context = zmq.Context()

            # create zmq sockets for handshake and data status update
            self.controller_handshake_socket = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_manager_id}-controller_handshake_socket-{uuid4().hex[:8]}".encode(),
            )
            self.data_status_update_socket = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_manager_id}-data_status_update_socket-{uuid4().hex[:8]}".encode(),
            )
            self.data_status_update_socket.connect(self.controller_info.to_addr("data_status_update_socket"))

            # do handshake with controller
            self._do_handshake_with_controller()

        except Exception as e:
            logger.error(f"Failed to connect to controller: {e}")
            raise

    def _do_handshake_with_controller(self) -> None:
        """Handshake with controller to establish connection with retransmission mechanism."""
        is_connected: bool = False
        pending_connection: bool = True
        handshake_retries: int = 0

        # Create zmq poller for handshake confirmation between controller and storage manager
        poller = zmq.Poller()

        self.controller_handshake_socket.connect(self.controller_info.to_addr("handshake_socket"))
        logger.debug(
            f"[{self.storage_manager_id}]: Handshake connection from storage manager id #{self.storage_manager_id} "
            f"to controller id #{self.controller_info.id} establish successfully."
        )
        poller.register(self.controller_handshake_socket, zmq.POLLIN)

        # Initial handshake request send
        self._send_handshake_requests()

        start_time = time.time()
        last_retry_time = time.time()

        while (
            not is_connected  # Only one controller to connect to
            and time.time() - start_time < TQ_STORAGE_HANDSHAKE_TIMEOUT
        ):
            # Check for timeout and retransmission
            current_time = time.time()
            if pending_connection:
                if (
                    current_time - last_retry_time >= TQ_STORAGE_HANDSHAKE_RETRY_INTERVAL
                    and handshake_retries < TQ_STORAGE_HANDSHAKE_MAX_RETRIES
                ):
                    logger.warning(
                        f"[{self.storage_manager_id}]: Retransmitting handshake "
                        f"to controller {self.controller_info.id}, "
                        f"attempt {handshake_retries + 1}/{TQ_STORAGE_HANDSHAKE_MAX_RETRIES}"
                    )
                    self._send_handshake_requests()
                    last_retry_time = current_time
                    handshake_retries += 1
                elif handshake_retries >= TQ_STORAGE_HANDSHAKE_MAX_RETRIES:
                    raise TimeoutError(
                        f"[{self.storage_manager_id}]: Handshake with controller {self.controller_info.id} "
                        f"({self.controller_info.ip}) failed after "
                        f"{TQ_STORAGE_HANDSHAKE_MAX_RETRIES} attempts."
                    )

            # Use shorter poll timeout for more responsive retry timing
            # while maintaining overall handshake timeout behavior
            poll_timeout = min(TQ_STORAGE_POLLER_TIMEOUT * 1000, 500)  # Max 500ms
            socks = dict(poller.poll(poll_timeout))

            if (socks.get(self.controller_handshake_socket, 0) & zmq.POLLIN) and pending_connection:
                try:
                    response_msg = ZMQMessage.deserialize(self.controller_handshake_socket.recv())

                    if response_msg.request_type == ZMQRequestType.HANDSHAKE_ACK:
                        is_connected = True
                        pending_connection = False
                        logger.debug(
                            f"[{self.storage_manager_id}]: Get handshake ACK response from "
                            f"controller id #{str(response_msg.sender_id)} to storage manager id "
                            f"#{self.storage_manager_id} successfully."
                        )
                except Exception as e:
                    logger.warning(
                        f"[{self.storage_manager_id}]: Error receiving handshake "
                        f"response from {self.controller_info.id}: {e}"
                    )

    def _send_handshake_requests(self) -> None:
        """Send handshake request to controller."""
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.HANDSHAKE,
            sender_id=self.storage_manager_id,
            body={
                "storage_manager_id": self.storage_manager_id,
                "storage_manager_type": self.__class__.__name__,
            },
        ).serialize()

        self.controller_handshake_socket.send(request_msg)
        logger.debug(
            f"[{self.storage_manager_id}]: Send handshake request from storage manager id "
            f"{self.storage_manager_id} to controller id #{self.controller_info.id} successfully."
        )

    async def notify_data_update(
        self,
        fields: list[str],
        global_indexes: list[int],
        dtypes: dict[int, dict[str, Any]],
        shapes: dict[int, dict[str, Any]],
    ) -> None:
        """
        Notify controller that new data is ready.

        Args:
            fields: Data update related fields.
            global_indexes: Data update related global_indexes.
            dtypes: Per-field dtypes for each field, in {global_index: {field: dtype}} format.
            shapes: Per-field shapes for each field, in {global_index: {field: shape}} format.
        """
        # Create zmq poller for notifying data update information

        if not self.controller_info:
            logger.warning(f"No controller connected for storage manager {self.storage_manager_id}")
            return

        # Create zmq poller for notifying data update information
        poller = zmq.Poller()
        # Note: data_status_update_socket is already connected during initialization

        try:
            poller.register(self.data_status_update_socket, zmq.POLLIN)

            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.NOTIFY_DATA_UPDATE,
                sender_id=self.storage_manager_id,
                body={
                    "fields": fields,
                    "global_indexes": global_indexes,
                    "dtypes": dtypes,
                    "shapes": shapes,
                },
            ).serialize()

            self.data_status_update_socket.send(request_msg)
            logger.debug(
                f"[{self.storage_manager_id}]: Send data status update request "
                f"from storage manager id #{self.storage_manager_id} "
                f"to controller id #{self.controller_info.id} successfully."
            )
        except Exception as e:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR,
                sender_id=self.storage_manager_id,
                body={
                    "message": f"Failed to notify data status update information from "
                    f"storage manager id #{self.storage_manager_id}, "
                    f"detail error message: {str(e)}"
                },
            ).serialize()

            self.data_status_update_socket.send(request_msg)

        # Make sure controller successfully receives data status update information.
        response_received: bool = False
        start_time = time.time()

        while (
            not response_received  # Only one controller to get response from
            and time.time() - start_time < TQ_DATA_UPDATE_RESPONSE_TIMEOUT
        ):
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT * 1000))

            if self.data_status_update_socket in socks:
                response_msg = ZMQMessage.deserialize(self.data_status_update_socket.recv())

                if response_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                    response_received = True
                    logger.debug(
                        f"[{self.storage_manager_id}]: Get data status update ACK response "
                        f"from controller id #{response_msg.sender_id} "
                        f"to storage manager id #{self.storage_manager_id} successfully."
                    )

        if not response_received:
            logger.error(
                f"[{self.storage_manager_id}]: Storage manager id #{self.storage_manager_id} "
                f"did not receive data status update ACK response from controller."
            )

    @abstractmethod
    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        raise NotImplementedError("Subclasses must implement put_data")

    @abstractmethod
    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        raise NotImplementedError("Subclasses must implement get_data")

    @abstractmethod
    async def clear_data(self, metadata: BatchMeta) -> None:
        raise NotImplementedError("Subclasses must implement clear_data")


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


# TODO (TQStorage): to be optimized. Now there are too many data dicts.
# transfer_data, transfer_dict, data, StorageUnitData, field_data...
def _add_field_data(
    transfer_dict: dict[str, Any], storage_meta_group: StorageMetaGroup, data: TensorDict
) -> dict[str, Any]:
    """Extract field data from TensorDict using sample_meta.batch_index as index.

    This function bridges the gap between raw TensorDict data and the transfer format
    needed for storage operations. The transfer_dict contains metadata and structure
    information, while the 'data' parameter contains the actual tensor values.

    Key Concept: sample_meta.batch_index represents the position of each sample's data
    in the original TensorDict (received from client). This function uses batch_index
    to extract the correct data items for each sample in the storage_meta_group.

    Args:
        transfer_dict: Dictionary containing transfer metadata with structure like:
            {
                "batch_indexes": [2, 0, 3],      # Positions in original TensorDict
                "global_indexes": [10, 11, 12],    # Global identifiers
                "local_indexes": [4, 5, 6],        # Storage locations
                "fields": ["field1", "field2"],
                "field_data": {}  # Will be populated by this function
            }
        storage_meta_group: StorageMetaGroup containing SampleMeta objects with:
            - sample_meta.batch_index: Position in original TensorDict
            - sample_meta.local_index: Position in storage unit
        data: Raw TensorDict with actual data (as received from client):
            TensorDict({"field1": [t0, t1, t2, t3, t4], "field2": [t5, t6, t7, t8, t9]})

    Returns:
        Updated transfer dictionary with field_data populated:
            {
                "batch_indexes": [2, 0, 3],
                "global_indexes": [10, 11, 12],
                "local_indexes": [4, 5, 6],
                "fields": ["field1", "field2"],
                "field_data": {
                    "field1": [t2, t0, t3],  # Extracted by batch_index from original data
                    "field2": [t7, t5, t8]
                }
            }

    Example:
        >>> # Raw data from client (TensorDict index 0-4)
        >>> data = TensorDict({"images": [img0, img1, img2, img3, img4]})
        >>> # storage_meta_group contains samples with batch_index [2, 0, 3]
        >>> transfer_dict = {
        ...     "fields": ["images"],
        ...     "batch_indexes": [2, 0, 3],
        ...     "local_indexes": [4, 5, 6],
        ...     "field_data": {}
        ... }
        >>> meta_group = StorageMetaGroup("storage1")
        >>> meta_group.add_sample_meta(SampleMeta(batch_index=2), 4)  # Extract img2
        >>> meta_group.add_sample_meta(SampleMeta(batch_index=0), 5)  # Extract img0
        >>> meta_group.add_sample_meta(SampleMeta(batch_index=3), 6)  # Extract img3
        >>> result = _add_field_data(transfer_dict, meta_group, data)
        >>> result["field_data"]["images"]  # [img2, img0, img3] - extracted by batch_index
    """
    field_names = transfer_dict["fields"]
    for fname in field_names:
        if fname in data.keys():
            index = [sample_meta.batch_index for sample_meta in storage_meta_group.sample_metas]

            result = itemgetter(*index)(data[fname])
            if not isinstance(result, tuple):
                result = (result,)
            transfer_dict["field_data"][fname] = list(result)

    return transfer_dict


def get_transfer_data(
    storage_meta_group: StorageMetaGroup,
    data: TensorDict,
) -> dict[str, Any]:
    """Convert StorageMetaGroup and TensorDict to transfer format for put operations.

    This function creates a bridge between the high-level metadata (StorageMetaGroup)
    and the raw data (TensorDict), producing a transfer_dict that contains both
    metadata structure and the actual field data needed for storage operations.

    Key Data Flow:
    1. storage_meta_group.get_transfer_data() creates metadata structure
    2. _add_field_data() extracts data using sample_meta.batch_index as key
    3. Final transfer_dict contains both metadata and correctly ordered data

    Args:
        storage_meta_group: StorageMetaGroup containing SampleMeta objects with:
            - sample_meta.batch_index: Position in original TensorDict (0-based)
            - sample_meta.global_index: Global unique identifier
            - sample_meta.local_index: Position in target storage unit
        data: Raw TensorDict with actual data values (as received from client):
            Format: {"field_name": [data_at_index_0, data_at_index_1, ...]}

    Returns:
        Complete transfer dictionary ready for storage operations:
            {
                "batch_indexes": [2, 0, 3],      # Original TensorDict positions
                "global_indexes": [10, 11, 12],    # Global identifiers
                "local_indexes": [4, 5, 6],        # Storage locations
                "fields": ["images", "labels"],
                "field_data": {
                    "images": [img2, img0, img3],  # Extracted by batch_index
                    "labels": [label2, label0, label3]
                }
            }

    Example:
        >>> # Client data: TensorDict with 5 samples (indices 0-4)
        >>> data = TensorDict({
        ...     "images": [img0, img1, img2, img3, img4],
        ...     "labels": [label0, label1, label2, label3, label4]
        ... })
        >>> # MetaGroup contains samples at positions 2, 0, 3 in original data
        >>> group = StorageMetaGroup("storage1")
        >>> group.add_sample_meta(SampleMeta(batch_index=2, global_index=10), 4)
        >>> group.add_sample_meta(SampleMeta(batch_index=0, global_index=11), 5)
        >>> group.add_sample_meta(SampleMeta(batch_index=3, global_index=12), 6)
        >>> transfer_dict = get_transfer_data(group, data)
        >>> transfer_dict["batch_indexes"]   # [2, 0, 3] - positions in original TensorDict
        >>> transfer_dict["field_data"]["images"]  # [img2, img0, img3] - extracted data

    Note:
        The critical insight is that sample_meta.batch_index is used to index into
        the original TensorDict to extract the correct data items. This ensures that
        even when samples are reordered or distributed across storage units,
        each sample's data is correctly mapped to its metadata.
    """

    result = storage_meta_group.get_transfer_data(field_names=list(data.keys()))
    result = _add_field_data(result, storage_meta_group, data)
    return result


def build_storage_meta_groups(
    batch_meta: BatchMeta,
    global_index_storage_unit_mapping: Callable,
    global_index_local_index_mapping: Callable,
) -> dict[str, StorageMetaGroup]:
    """Build storage meta groups from batch metadata for distributed storage.

    This function is the starting point of the data distribution workflow. It analyzes
    BatchMeta containing SampleMeta objects (originating from client requests) and
    groups them by target storage unit based on their global_index.

    Key Data Flow:
    1. BatchMeta contains SampleMeta objects with batch_index (original TensorDict position)
    2. Each SampleMeta is assigned to a storage unit using global_index mapping
    3. Local storage positions are calculated for each sample
    4. Results in StorageMetaGroup objects ready for transfer operations

    Args:
        batch_meta: BatchMeta containing SampleMeta objects from client request.
            Each SampleMeta has:
            - batch_index: Position in original TensorDict (0-based)
            - global_index: Global unique identifier across all storage
        global_index_storage_unit_mapping: Function to map global_index to storage_unit_id.
            Example: lambda x: storage_unit_ids[x % num_storage_units] (round-robin distribution)
        global_index_local_index_mapping: Function to map global_index to local_index.
            Example: lambda x: x // num_storage_units (local position within storage unit)

    Returns:
        Dictionary mapping storage_unit_id to StorageMetaGroup, where each group contains:
        - storage_id: Target storage unit identifier
        - sample_metas: List of SampleMeta objects assigned to this unit
        - local_indexes: List of storage positions for each sample

    Example:
        >>> # Input: BatchMeta with samples at global_indexes [10, 11, 12]
        >>> # 3 storage units available: storage_0, storage_1, storage_2
        >>> batch_meta = BatchMeta(samples=[
        ...     SampleMeta(batch_index=0, global_index=10),  # Original position 0
        ...     SampleMeta(batch_index=1, global_index=11),  # Original position 1
        ...     SampleMeta(batch_index=2, global_index=12)   # Original position 2
        ... ])
        >>> groups = build_storage_meta_groups(
        ...     batch_meta,
        ...     lambda x: f"storage_{x % 3}",  # 10->storage_1, 11->storage_2, 12->storage_0
        ...     lambda x: x // 3               # 10->3, 11->3, 12->4
        ... )
        >>> groups["storage_1"].sample_metas[0].batch_index  # 0 - original TensorDict position
        >>> groups["storage_1"].sample_metas[0].local_index  # 3 - storage position

    Note:
        This function preserves the crucial batch_index information that links each
        SampleMeta back to its original position in the client's TensorDict.
        This batch_index is later used by _add_field_data() to extract
        the correct data items for storage.
    """
    storage_meta_groups: dict[str, StorageMetaGroup] = {}

    for sample in batch_meta.samples:
        storage_id = global_index_storage_unit_mapping(sample.global_index)
        local_index = global_index_local_index_mapping(sample.global_index)
        if storage_id not in storage_meta_groups:
            storage_meta_groups[storage_id] = StorageMetaGroup(storage_id=storage_id)

        # Use add_sample_meta to store SampleMeta references directly
        storage_meta_groups[storage_id].add_sample_meta(sample, local_index)

    return storage_meta_groups


class AsyncSimpleStorageManager(TransferQueueStorageManager):
    """Asynchronous storage manager that handles multiple storage units.

    This manager provides async put/get/clear operations across multiple SimpleStorageUnit
    instances using ZMQ communication and dynamic socket management.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        self.config = config
        self.storage_unit_infos = config.get("storage_unit_infos", None)  # type: ZMQServerInfo | dict[str, ZMQServerInfo]
        assert self.storage_unit_infos is not None

        self.num_storage_units = len(self.storage_unit_infos)
        self._build_storage_mapping_functions()

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_storage_units)

    def _build_storage_mapping_functions(self):
        """Build mapping functions for global index to storage unit and local index.

        Creates round-robin mapping functions to distribute data across storage units.
        """
        self.global_index_storage_unit_mapping = lambda x: list(self.storage_unit_infos.keys())[
            x % self.num_storage_units
        ]
        self.global_index_local_index_mapping = lambda x: x // self.num_storage_units

    def _register_servers(self, server_infos: "ZMQServerInfo | dict[Any, ZMQServerInfo]"):
        """Register and validate server information.

        Args:
            server_infos: ZMQServerInfo | dict[Any, ZMQServerInfo])
                ZMQServerInfo or dict of server infos to register.

        Returns:
            Dictionary with server IDs as keys and ZMQServerInfo objects as values.

        Raises:
            ValueError: If server_infos format is invalid.
        """
        server_infos_transform = {}
        if isinstance(server_infos, ZMQServerInfo):
            server_infos_transform[server_infos.id] = server_infos
        elif isinstance(server_infos, dict):
            for k, v in server_infos.items():
                if not isinstance(v, ZMQServerInfo):
                    raise ValueError(f"Invalid server info for key {k}: {v}")
                server_infos_transform[v.id] = v
        else:
            raise ValueError(f"Invalid server infos: {server_infos}")

        return server_infos_transform

    # TODO (TQStorage): Provide a general dynamic socket function for both Client & Storage @huazhong.
    @staticmethod
    def dynamic_storage_manager_socket(socket_name: str):
        """Decorator to auto-manage ZMQ sockets for Controller/Storage servers (create -> connect -> inject -> close).

        Args:
            socket_name (str): Port name (from server config) to use for ZMQ connection (e.g., "data_req_port").

        Decorated Function Rules:
            1. Must be an async class method (needs `self`).
            2. `self` requires:
            - `storage_unit_infos: storage unit infos (ZMQServerInfo | dict[Any, ZMQServerInfo]).
            3. Specify target server via:
            - `target_stroage_unit` arg.
            4. Receives ZMQ socket via `socket` keyword arg (injected by decorator).
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                server_key = kwargs.get("target_storage_unit")
                if server_key is None:
                    for arg in args:
                        if isinstance(arg, str) and arg in self.storage_unit_infos.keys():
                            server_key = arg
                            break

                server_info = self.storage_unit_infos.get(server_key)

                if not server_info:
                    raise RuntimeError(f"Server {server_key} not found in registered servers")

                context = zmq.asyncio.Context()
                address = f"tcp://{server_info.ip}:{server_info.ports.get(socket_name)}"
                identity = f"{self.storage_manager_id}_to_{server_info.id}_{uuid4()}".encode()
                sock = create_zmq_socket(context, zmq.DEALER, identity=identity)

                try:
                    sock.connect(address)
                    # Timeouts to avoid indefinite await on recv/send
                    sock.setsockopt(zmq.RCVTIMEO, 10_000)  # 10s
                    sock.setsockopt(zmq.SNDTIMEO, 10_000)  # 10s
                    logger.info(
                        f"[{self.storage_manager_id}]: Connected to StorageUnit {server_info.id} at {address} "
                        f"with identity {identity.decode()}"
                    )

                    kwargs["socket"] = sock
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"[{self.storage_manager_id}]: Error in socket operation with StorageUnit {server_info.id}: {e}"
                    )
                    raise
                finally:
                    try:
                        if not sock.closed:
                            sock.setsockopt(zmq.LINGER, 0)
                            sock.close()
                    except Exception as e:
                        logger.warning(
                            f"[{self.storage_manager_id}]: Error closing socket to StorageUnit {server_info.id}: {e}"
                        )

                    context.term()

            return wrapper

        return decorator

    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """
        Send data to remote StorageUnit based on metadata.

        Args:
            data: TensorDict containing the data to store.
            metadata: BatchMeta containing storage location information.
        """

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(
            metadata, self.global_index_storage_unit_mapping, self.global_index_local_index_mapping
        )

        # send data to each storage unit using parallel threads for each meta_group
        def put_data_single(storage_id, meta_group):
            return asyncio.run(
                self._put_to_single_storage_unit(get_transfer_data(meta_group, data), target_storage_unit=storage_id)
            )

        futures = [
            self._thread_pool.submit(put_data_single, storage_id, meta_group)
            for storage_id, meta_group in storage_meta_groups.items()
        ]
        concurrent.futures.wait(futures)
        for future in futures:
            future.result()

        # Gather per-field dtype and shape information for each field
        # global_indexes, local_indexes, and field_data correspond one-to-one
        per_field_dtypes = {}
        per_field_shapes = {}

        # Initialize the data structure for each global index
        for global_idx in metadata.global_indexes:
            per_field_dtypes[global_idx] = {}
            per_field_shapes[global_idx] = {}

        # For each field, extract dtype and shape for each sample
        for field in data.keys():
            for i, data_item in enumerate(data[field]):
                global_idx = metadata.global_indexes[i]
                per_field_dtypes[global_idx][field] = data_item.dtype if hasattr(data_item, "dtype") else None
                per_field_shapes[global_idx][field] = data_item.shape if hasattr(data_item, "shape") else None

        # notify controller that new data is ready
        await self.notify_data_update(list(data.keys()), metadata.global_indexes, per_field_dtypes, per_field_shapes)

    @dynamic_storage_manager_socket(socket_name="put_get_socket")
    async def _put_to_single_storage_unit(self, transfer_data: dict[str, Any], target_storage_unit=None, socket=None):
        """
        Send data to a specific storage unit.
        """
        local_indexes = transfer_data["local_indexes"]

        tensordict_data = TensorDict(
            {
                field: (
                    torch.nested.as_nested_tensor(transfer_data["field_data"][field])
                    if transfer_data["field_data"][field]
                    and all(isinstance(x, torch.Tensor) for x in transfer_data["field_data"][field])
                    else NonTensorStack(*transfer_data["field_data"][field])
                )
                for field in transfer_data["field_data"]
            }
        )

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"local_indexes": local_indexes, "data": tensordict_data},
        )

        try:
            await socket.send(request_msg.serialize())
            serialized = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized)

            if response_msg.request_type != ZMQRequestType.PUT_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to put data to storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error in put to storage unit {target_storage_unit}: {str(e)}") from e

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        """
        Retrieve data from remote StorageUnit based on metadata.

        Args:
            metadata: BatchMeta that contains metadata for data retrieval.

        Returns:
            TensorDict containing the retrieved data.
        """

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(
            metadata, self.global_index_storage_unit_mapping, self.global_index_local_index_mapping
        )

        # retrieve data using parallel threads for each meta_group
        def get_data_single(storage_id, meta_group):
            return asyncio.run(
                self._get_from_single_storage_unit(meta_group.get_transfer_data(), target_storage_unit=storage_id)
            )

        futures = [
            self._thread_pool.submit(get_data_single, storage_id, meta_group)
            for storage_id, meta_group in storage_meta_groups.items()
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # post-process data segments to generate a batch of data
        merged_data: dict[int, dict[str, torch.Tensor]] = {}
        for global_indexes, fields, data_from_single_storage_unit in results:
            field_getter = itemgetter(*fields)
            field_values = field_getter(data_from_single_storage_unit)

            if len(fields) == 1:
                extracted_data = {fields[0]: field_values}
            else:
                extracted_data = dict(zip(fields, field_values, strict=False))

            for idx, global_idx in enumerate(global_indexes):
                if global_idx not in merged_data:
                    merged_data[global_idx] = {}
                merged_data[global_idx].update({field: extracted_data[field][idx] for field in fields})

        ordered_data: dict[str, list[torch.Tensor]] = {}
        for field in metadata.field_names:
            ordered_data[field] = [merged_data[global_idx][field] for global_idx in metadata.global_indexes]

        with limit_pytorch_auto_parallel_threads():
            tensor_data = {
                field: (
                    torch.stack(torch.nested.as_nested_tensor(v).unbind())
                    if v
                    and all(isinstance(item, torch.Tensor) for item in v)
                    and all(item.shape == v[0].shape for item in v)
                    else (
                        torch.nested.as_nested_tensor(v)
                        if v and all(isinstance(item, torch.Tensor) for item in v)
                        else NonTensorStack(*v)
                    )
                )
                for field, v in ordered_data.items()
            }

        return TensorDict(tensor_data, batch_size=len(metadata))

    @dynamic_storage_manager_socket(socket_name="put_get_socket")
    async def _get_from_single_storage_unit(self, index_data, target_storage_unit=None, socket=None):
        global_indexes = index_data["global_indexes"]
        local_indexes = index_data["local_indexes"]
        fields = index_data["fields"]

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"local_indexes": local_indexes, "fields": fields},
        )

        try:
            await socket.send(request_msg.serialize())
            serialized = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized)
            logger.info(
                f"[{self.storage_manager_id}]: get data response from storage unit "
                f"{target_storage_unit}: {response_msg}"
            )

            if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE:
                # Return data and index information from this storage unit
                storage_unit_data = response_msg.body["data"]
                return global_indexes, fields, storage_unit_data
            else:
                raise RuntimeError(
                    f"Failed to get data from storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error getting data from storage unit {target_storage_unit}: {str(e)}") from e

    async def clear_data(self, metadata: BatchMeta) -> None:
        """Clear data in remote StorageUnit.

        Args:
            metadata: BatchMeta that contains metadata for data clearing.
        """

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(
            metadata, self.global_index_storage_unit_mapping, self.global_index_local_index_mapping
        )

        # clear data
        tasks = [
            self._clear_single_storage_unit(
                meta_group.get_transfer_data()["local_indexes"], target_storage_unit=storage_id
            )
            for storage_id, meta_group in storage_meta_groups.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[{self.storage_manager_id}]: Error in clear operation task {i}: {result}")

    @dynamic_storage_manager_socket(socket_name="put_get_socket")
    async def _clear_single_storage_unit(self, local_indexes, target_storage_unit=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA,
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"local_indexes": local_indexes},
            )

            await socket.send(request_msg.serialize())
            serialized_msg = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_msg)

            if response_msg.request_type != ZMQRequestType.CLEAR_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear storage {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(f"[{self.storage_manager_id}]: Successfully clear storage unit {target_storage_unit}")
        except Exception as e:
            logger.error(f"[{self.storage_manager_id}]: Error clearing storage unit {target_storage_unit}: {str(e)}")
            raise

    def get_zmq_server_info(self) -> dict[str, ZMQServerInfo]:
        """Get ZMQ server information for all storage units.

        Returns:
            Dictionary mapping storage unit IDs to their ZMQServerInfo.
        """
        return self.storage_unit_infos

    def cleanup(self) -> None:
        """Clean up resources including thread pool."""
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=True)

    def close(self) -> None:
        """Close all ZMQ sockets and context to prevent resource leaks."""
        for sock in (
            self.controller_handshake_socket,
            self.data_status_update_socket,
            getattr(self, "put_get_socket", None),
        ):
            try:
                if sock and not sock.closed:
                    sock.setsockopt(zmq.LINGER, 0)
                    sock.close()
            except Exception:
                pass
        try:
            if hasattr(self, "zmq_context") and self.zmq_context:
                self.zmq_context.term()
        except Exception:
            pass

    def __del__(self):
        """Destructor to ensure thread pool is cleaned up."""
        try:
            self.cleanup()
        except Exception:
            pass


class TransferQueueStorageManagerFactory:
    """Factory that creates a StorageManager instance."""

    _registry: dict[str, type[TransferQueueStorageManager]] = {}

    @classmethod
    def register(cls, manager_type: str, manager_cls: type[TransferQueueStorageManager]):
        if not issubclass(manager_cls, TransferQueueStorageManager):
            raise TypeError(f"manager_cls {type(manager_type)} must be a subclass of TransferQueueStorageManager")
        cls._registry[manager_type] = manager_cls

    @classmethod
    def create(cls, manager_type: str, config: dict[str, Any]) -> TransferQueueStorageManager:
        if manager_type not in cls._registry:
            raise ValueError(
                f"Unknown manager_type: {manager_type}. Supported managers include: {list(cls._registry.keys())}"
            )
        return cls._registry[manager_type](config)


# TODO (TQStorage): Provide a KVStorageManager implementation, where we can select different
#                   backend storages like MoonCake, Redis, etc.


# Register all the StorageManager
TransferQueueStorageManagerFactory.register("AsyncSimpleStorageManager", AsyncSimpleStorageManager)
