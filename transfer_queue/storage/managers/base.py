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

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import zmq
from tensordict import TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo, create_zmq_socket

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


class KVStorageManager(TransferQueueStorageManager):
    """
    A storage manager that uses a key-value (KV) backend (e.g., YuanRong) to store and retrieve tensor data.
    It maps structured metadata (BatchMeta) to flat lists of keys and values for efficient KV operations.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the KVStorageManager with configuration.
        """
        super().__init__(config)
        client_name = config.get("client_name", "Yuanrong")
        self.storage_client = StorageClientFactory.create(client_name, config)

    @staticmethod
    def _generate_yr_keys(metadata: BatchMeta) -> list[str]:
        """
        Generate KV keys in the format 'global_index@field_name' for all sample-field pairs.

        Keys are generated in sorted order by field name first, then by global index,
        ensuring consistent ordering for batched operations.

        Args:
            metadata (BatchMeta): Metadata containing global indexes and field names.

        Returns:
            list[str]: List of keys, e.g., ['0@field_a', '1@field_a', '0@field_b', ...]
        """
        return [
            f'{index}@{field}'
            for field in sorted(metadata.field_names)
            for index in metadata.global_indexes
        ]

    @staticmethod
    def _generate_yr_values(data: TensorDict) -> list[Tensor]:
        """
        Extract and flatten tensor values from a TensorDict in field-major order.

        Values are ordered by sorted field names, then by row (sample) order within each field.
        This matches the key order generated by `_generate_yr_keys`.

        Args:
            data (TensorDict): Input data where keys are field names and values are tensors.

        Raises:
            TypeError: If any value in the TensorDict is not a torch.Tensor.

        Returns:
            list[Tensor]: Flattened list of tensors, e.g.,
                          [data[field_a][0], data[field_a][1], data[field_a][2], ..., data[field_b][0], ...]
        """
        for v in data.values():
            if not torch.is_tensor(v):
                raise TypeError(f"TensorDict values must be torch.Tensor, but got {type(v)}")

        return [
            row_data
            for field in sorted(data.keys())
            for row_data in data[field]
        ]

    @staticmethod
    def _merge_kv_to_dict(metadata: BatchMeta, values: list[Tensor]) -> TensorDict:
        """
        Reconstruct a TensorDict from a list of values using metadata.

        The values list is assumed to be in the same order as keys generated by `_generate_yr_keys`.
        This method reshapes the flat list back into a structured TensorDict.

        Args:
            metadata (BatchMeta): Metadata containing global indexes and field names.
            values (list[Tensor]): List of tensors in field-major order.

        Raises:
            ValueError: If the length of values does not match expected (num_samples * num_fields).

        Returns:
            TensorDict: Reconstructed tensor dictionary with batch size equal to number of samples.
        """
        global_indexes = metadata.global_indexes
        field_names = sorted(metadata.field_names)
        expected_length = len(global_indexes) * len(field_names)
        if len(values) != expected_length:
            raise ValueError(f'Length of values ({len(values)}) does not match expected ({expected_length})')

        if len(values) == 0:
            return TensorDict({}, batch_size=len(global_indexes))

        merged_data: dict[str, list[Tensor]] = {field: [] for field in field_names}

        # Group values by field_name
        value_idx = 0
        for field in field_names:
            for _ in range(len(global_indexes)):
                merged_data[field].append(values[value_idx])
                value_idx += 1

        # Stack or nest tensors per field
        tensor_data = {}
        for field, tensor_list in merged_data.items():
            try:
                tensor_data[field] = torch.stack(tensor_list)
            except RuntimeError as re:
                # Fallback to nested tensor if shapes are irregular
                tensor_data[field] = torch.nested.as_nested_tensor(tensor_list)

        return TensorDict(tensor_data, batch_size=len(global_indexes))

    @staticmethod
    def _get_shape_type_list(metadata: BatchMeta):
        """
        Extract the expected shape and dtype for each field-sample pair in metadata.

        The order matches the key/value order: sorted by field name, then by global index.

        Args:
            metadata (BatchMeta): Metadata containing sample and field information.

        Returns:
            tuple[list[torch.Size], list[torch.dtype]]: Two lists containing the shape and dtype
            for each tensor to be retrieved.
        """
        shapes = []
        dtypes = []
        for field_name in sorted(metadata.field_names):
            for index in range(len(metadata)):
                field = metadata.samples[index].get_field_by_name(field_name)
                shapes.append(field.shape)
                dtypes.append(field.dtype)
        return shapes, dtypes

    # TODO: Test put_data/get_data/clear_data with YuanrongStorageClient
    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        keys = self._generate_yr_keys(metadata)
        values = self._generate_yr_values(data)
        self.storage_client.put(keys=keys, values=values)

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        keys = self._generate_yr_keys(metadata)
        shapes, dtypes = self._get_shape_type_list(metadata)
        values = self.storage_client.get(keys=keys, shapes=shapes, dtypes=dtypes)
        return self._merge_kv_to_dict(metadata, values)

    async def clear_data(self, metadata: BatchMeta) -> None:
        keys = self._generate_yr_keys(metadata)
        self.storage_client.clear(keys=keys)