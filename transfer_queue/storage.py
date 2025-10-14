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
import os
import time
from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
from functools import wraps
from threading import Thread
from typing import Any, Callable, Optional
from uuid import uuid4
from operator import itemgetter

import ray
import torch
import zmq
from ray.util import get_node_ip_address
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta, SampleMeta
from transfer_queue.utils.utils import TransferQueueRole
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

TQ_STORAGE_POLLER_TIMEOUT = os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 1000)
TQ_STORAGE_HANDSHAKE_TIMEOUT = int(os.environ.get("TQ_STORAGE_HANDSHAKE_TIMEOUT", 30))
TQ_DATA_UPDATE_RESPONSE_TIMEOUT = int(os.environ.get("TQ_DATA_UPDATE_RESPONSE_TIMEOUT", 600))


class StorageUnitData:
    """
    Class used for storing several elements, each element is composed of several fields and corresponding data, like:
    #####################################################
    # local_index | field_name1 | field_name2 | ...   #
    # 0           | item1       | item2       | ...   #
    # 1           | item3       | item4       | ...   #
    # 2           | item5       | item6       | ...   #
    #####################################################
    """

    def __init__(self, storage_size: int):
        # Dict containing field names and corresponding data in the field, e.g. {"field_name1": [data1, data2, ...]}
        self.field_data: dict[str, list] = {}

        # Maximum number of elements stored in storage unit
        self.storage_size = storage_size

    def get_data(self, fields: list[str], local_indexes: list[int]) -> TensorDict[str, list]:
        """
        Get data from storage unit according to given fields and local_indexes.

        param:
            fields: Field names used for getting data.
            local_indexes: Local indexes used for getting data.
        return:
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

        param:
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

        param:
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



class TransferQueueStorageManager(ABC):
    """Base class for storage layer. It defines the interface for data operation and
    general provide handshake & notification capabilities."""

    def __init__(self, config: dict[str, Any]):
        self.storage_manager_id = f"TQ_STORAGE_{uuid4().hex[:8]}"
        self.controller_infos = {} # type: dict[str, ZMQServerInfo]
        self.data_status_update_sockets = {}
        self.controller_handshake_sockets = {}
        self.config = config

        self.zmq_context = None
        self.connect_to_controllers(self.controller_infos)

    def connect_to_controllers(self, controller_infos: ZMQServerInfo | dict[str, ZMQServerInfo]) -> None:
        """Initialize ZMQ sockets between storage unit and controllers for handshake."""
        try:
            if isinstance(controller_infos, ZMQServerInfo):
                controller_infos = {controller_infos.id: controller_infos}

            self.controller_infos = controller_infos

            # create zmq context
            self.zmq_context = zmq.Context()

            # create zmq sockets for handshake and data status update
            for controller_id, controller_info in self.controller_infos.items():
                self.controller_handshake_sockets[controller_id] = create_zmq_socket(
                    self.zmq_context,
                    zmq.DEALER,
                    identity=f"{self.storage_manager_id}-controller_handshake_sockets-{uuid4().hex[:8]}".encode(),
                )
                self.data_status_update_sockets[controller_id] = create_zmq_socket(
                    self.zmq_context,
                    zmq.DEALER,
                    identity=f"{self.storage_manager_id}-data_status_update_sockets-{uuid4().hex[:8]}".encode(),
                )

            # do handshake with controllers
            self._do_handshake_with_controller()

        except Exception as e:
            logger.error(f"Failed to connect to controllers: {e}")

    def _do_handshake_with_controller(self) -> None:
        """Handshake with all controllers to establish connection."""
        connected_controllers: set[str] = set()

        # Create zmq poller for handshake confirmation between controller and storage manager
        poller = zmq.Poller()

        for controller_id, controller_info in self.controller_infos.items():
            self.controller_handshake_sockets[controller_id].connect(controller_info.to_addr("handshake_socket"))
            logger.debug(
                f"[{self.storage_manager_id}]: Handshake connection from storage manager id #{self.storage_manager_id} "
                f"to controller id #{controller_id} establish successfully."
            )

            # Send handshake request to controllers
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.HANDSHAKE,
                sender_id=self.storage_manager_id,
                body={
                    "storage_manager_id": self.storage_manager_id,
                    "storage_manager_type": self.__class__.__name__,
                },
            ).serialize()

            self.controller_handshake_sockets[controller_id].send(request_msg)
            logger.debug(
                f"[{self.storage_manager_id}]: Send handshake request from storage manager id #{self.storage_manager_id} "
                f"to controller id #{controller_id} successfully."
            )

            poller.register(self.controller_handshake_sockets[controller_id], zmq.POLLIN)

        start_time = time.time()
        while (
            len(connected_controllers) < len(self.controller_infos)
            and time.time() - start_time < TQ_STORAGE_HANDSHAKE_TIMEOUT
        ):
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT))

            for controller_handshake_socket in self.controller_handshake_sockets.values():
                if controller_handshake_socket in socks:
                    response_msg = ZMQMessage.deserialize(controller_handshake_socket.recv())

                    if response_msg.request_type == ZMQRequestType.HANDSHAKE_ACK:
                        connected_controllers.add(response_msg.sender_id)
                        logger.debug(
                            f"[{self.storage_manager_id}]: Get handshake ACK response from "
                            f"controller id #{str(response_msg.sender_id)} to storage manager id "
                            f"#{self.storage_manager_id} successfully."
                        )

        if len(connected_controllers) < len(self.controller_infos):
            logger.warning(
                f"[{self.storage_manager_id}]: Only get {len(connected_controllers)} / {len(self.controller_infos)} "
                f"successful handshake connections to controllers from storage manager id #{self.storage_manager_id}"
            )

    async def notify_data_update(
        self,
        fields: list[str],
        global_indexes: list[int],
        dtypes: dict[int, dict[str, Any]],
        shapes: dict[int, dict[str, Any]],
    ) -> None:
        """
        Broadcast data status update to all controllers.

        param:
            fields: data update related fields.
            global_indexes: data update related global_indexes.
            dtypes: per-field dtypes for each field, in {global_index: {field: dtype}} format.
            shapes: per-field shapes for each field, in {global_index: {field: shape}} format.
        """
        # Create zmq poller for notifying data update information

        if not self.controller_infos:
            logger.warning(f"No controllers connected for storage manager {self.storage_manager_id}")
            return

        # Create zmq poller for notifying data update information
        poller = zmq.Poller()

        # Connect data status update socket to all controllers
        for controller_id, controller_info in self.controller_infos.items():
            data_status_update_socket = self.data_status_update_sockets[controller_id]
            data_status_update_socket.connect(controller_info.to_addr("data_status_update_socket"))
            logger.debug(
                f"[{self.storage_manager_id}]: Data status update connection from "
                f"storage manager id #{self.storage_manager_id} to "
                f"controller id #{controller_id} establish successfully."
            )

            try:
                poller.register(data_status_update_socket, zmq.POLLIN)

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

                data_status_update_socket.send(request_msg)
                logger.debug(
                    f"[{self.storage_manager_id}]: Send data status update request "
                    f"from storage manager id #{self.storage_manager_id} "
                    f"to controller id #{controller_id} successfully."
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

                data_status_update_socket.send(request_msg)

        # Make sure all controllers successfully receive data status update information.
        response_controllers: set[str] = set()
        start_time = time.time()

        while (
            len(response_controllers) < len(self.controller_infos)
            and time.time() - start_time < TQ_DATA_UPDATE_RESPONSE_TIMEOUT
        ):
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT))

            for data_status_update_socket in self.data_status_update_sockets.values():
                if data_status_update_socket in socks:
                    response_msg = ZMQMessage.deserialize(data_status_update_socket.recv())

                    if response_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                        response_controllers.add(response_msg.sender_id)
                        logger.debug(
                            f"[{self.storage_manager_id}]: Get data status update ACK response "
                            f"from controller id #{response_msg.sender_id} "
                            f"to storage manager id #{self.storage_manager_id} successfully."
                        )

        if len(response_controllers) < len(self.controller_infos):
            logger.warning(
                f"[{self.storage_manager_id}]: Storage manager id #{self.storage_manager_id} "
                f"only get {len(response_controllers)} / {len(self.controller_infos)} "
                f"data status update ACK responses from controllers."
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


@ray.remote(num_cpus=1)
class TransferQueueStorageSimpleUnit:
    def __init__(self, storage_unit_size: int):
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4().hex[:8]}"
        self.storage_unit_size = storage_unit_size

        self.storage_data = StorageUnitData(self.storage_unit_size)

        self.zmq_server_info = ZMQServerInfo.create(
            role=TransferQueueRole.STORAGE,
            id=str(self.storage_unit_id),
            ip=get_node_ip_address(),
            ports={"put_get_socket": get_free_port()},
        )
        self._init_zmq_socket()
        self._start_process_put_get()

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controllers/clients:
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
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT))

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

        param:
            data_parts: ZMQMessage from client.
        return:
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

        param:
            data_parts: ZMQMessage from client.
        return:
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

        param:
            data_parts: ZMQMessage from client, including target local_indexes.
        return:
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
        return self.zmq_server_info


class AsyncTransferQueueStorageSimpleUnitManager(TransferQueueStorageManager):
    def __init__(self, storage_unit_infos: ZMQServerInfo | dict[str, ZMQServerInfo], config: dict[str, Any]):
        super().__init__(config)
        self.storage_unit_infos = self._register_servers(storage_unit_infos)
        self.storage_unit_size = config.get("storage_unit_size", 10000)

        self._build_storage_mapping_functions()


    def _build_storage_mapping_functions(self):
        self.global_index_storage_unit_mapping = lambda x: list(self.storage_unit_infos.keys())[x % len(self.storage_unit_infos)]
        self.global_index_local_index_mapping = lambda x: x // len(self.storage_unit_infos)


    def _register_servers(self, server_infos):
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
                            sock.setsockopt(zmq.LINGER, -1)
                            sock.close()
                        sock.close(linger=0)
                    except Exception as e:
                        logger.warning(f"[{self.storage_manager_id}]: Error closing socket to StorageUnit {server_info.id}: {e}")

                    context.term()

            return wrapper

        return decorator

    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """
        Send data to remote StorageUnit based on metadata.
        """

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(metadata, self.global_index_storage_unit_mapping,
                                                        self.global_index_local_index_mapping)

        # send data to each storage unit
        tasks = [
            self._put_to_single_storage_unit(get_transfer_data(meta_group, data), target_storage_unit=storage_id)
            for storage_id, meta_group in storage_meta_groups.items()
        ]
        await asyncio.gather(*tasks)

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

        # notify all controllers that new data is ready
        await self.notify_data_update(list(data.keys()), metadata.global_indexes, per_field_dtypes, per_field_shapes)

    @dynamic_storage_manager_socket(socket_name="put_get_socket")
    async def _put_to_single_storage_unit(
        self, transfer_data: dict[str, Any], target_storage_unit=None, socket=None
    ):
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
        """

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(metadata, self.global_index_storage_unit_mapping,
                                                        self.global_index_local_index_mapping)

        # retrive data
        tasks = [
            self._get_from_single_storage_unit(meta_group.get_transfer_data(), target_storage_unit=storage_id)
            for storage_id, meta_group in storage_meta_groups.items()
        ]

        results = await asyncio.gather(*tasks)

        # post-process data segments to generate a batch of data
        merged_data: dict[int, dict[str, torch.Tensor]] = {}
        for global_indexes, fields, data_from_single_storage_unit in results:
            extracted_data = {field: data_from_single_storage_unit[field] for field in fields}

            for idx, global_idx in enumerate(global_indexes):
                if global_idx not in merged_data:
                    merged_data[global_idx] = {}
                for field in fields:
                    merged_data[global_idx][field] = extracted_data[field][idx]

        ordered_data: dict[str, list[torch.Tensor]] = {field: [] for field in metadata.field_names}
        for global_idx in metadata.global_indexes:
            for field in metadata.field_names:
                ordered_data[field].append(merged_data[global_idx][field])

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
        tensor_data["global_indexes"] = torch.tensor(metadata.global_indexes)

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
            logger.info(f"[{self.storage_manager_id}]: get data response from storage unit {target_storage_unit}: {response_msg}")

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
        """Clear data in remote StorageUnit"""

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(metadata, self.global_index_storage_unit_mapping,
                                                        self.global_index_local_index_mapping)

        # clear data
        tasks = [
            self._clear_single_storage_unit(meta_group.get_transfer_data()["local_indexes"], target_storage_unit=storage_id)
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
                    f"Failed to clear storage {target_storage_unit}: {response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(f"[{self.storage_manager_id}]: Successfully clear storage unit {target_storage_unit}")
        except Exception as e:
            logger.error(f"[{self.storage_manager_id}]: Error clearing storage unit {target_storage_unit}: {str(e)}")
            raise

    def get_zmq_server_info(self) -> dict[str, ZMQServerInfo]:
        return self.storage_unit_infos


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
        """Convert to dictionary format for backward compatibility"""
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

# TODO: to be optimized. Now there are too many data dicts.
# transfer_data, transfer_dict, data, StorageUnitData, field_data...
def _add_field_data(
    transfer_dict: dict[str, Any], storage_meta_group: StorageMetaGroup, data: TensorDict
) -> dict[str, Any]:
    """Helper function to add field data to the transfer dictionary"""
    field_names = transfer_dict["fields"]
    for fname in field_names:
        if fname in data.keys():
            transfer_dict["field_data"][fname] = []
            for sample_meta in storage_meta_group.sample_metas:
                transfer_dict["field_data"][fname].append(data[fname][sample_meta.batch_index])
    return transfer_dict


def get_transfer_data(
    storage_meta_group: StorageMetaGroup,
    data: TensorDict,
) -> dict[str, Any]:
    """Convert to dictionary format with field data for put operations"""

    result = storage_meta_group.get_transfer_data(field_names=list(data.keys()))
    result = _add_field_data(result, storage_meta_group, data)
    return result


def build_storage_meta_groups(
        batch_meta: BatchMeta,
        global_index_storage_unit_mapping: Callable,
        global_index_local_index_mapping: Callable,
) -> dict[str, StorageMetaGroup]:
    """Build storage groups from samples during initialization"""
    storage_meta_groups: dict[str, StorageMetaGroup] = {}

    for sample in batch_meta.samples:
        storage_id = global_index_storage_unit_mapping(sample.global_index)
        local_index = global_index_local_index_mapping(sample.global_index)
        if storage_id not in storage_meta_groups:
            storage_meta_groups[storage_id] = StorageMetaGroup(storage_id=storage_id)

        # Use add_sample_meta to store SampleMeta references directly
        storage_meta_groups[storage_id].add_sample_meta(sample, local_index)

    return storage_meta_groups
