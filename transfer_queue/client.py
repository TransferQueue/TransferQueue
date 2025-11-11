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
from functools import wraps
from typing import Any, Callable, Optional, Union
from uuid import uuid4

import ray
import zmq
import zmq.asyncio
from tensordict import TensorDict

from transfer_queue.controller import TransferQueueController
from transfer_queue.metadata import (
    BatchMeta,
)
from transfer_queue.storage import (
    SimpleStorageUnit,
    TransferQueueStorageManager,
    TransferQueueStorageManagerFactory,
)
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


class AsyncTransferQueueClient:
    """Asynchronous client for interacting with TransferQueue controller and storage systems.

    This client provides async methods for data transfer operations including getting metadata,
    reading data from storage, writing data to storage, and clearing data.
    """

    def __init__(
        self,
        client_id: str,
        controller_info: ZMQServerInfo,
    ):
        """Initialize the asynchronous TransferQueue client.

        Args:
            client_id: Unique identifier for this client instance
            controller_info: Single controller ZMQ server information
        """
        if controller_info is None:
            raise ValueError("controller_info cannot be None")
        if not isinstance(controller_info, ZMQServerInfo):
            raise TypeError(f"controller_info must be ZMQServerInfo, got {type(controller_info)}")
        self.client_id = client_id
        self._controller: ZMQServerInfo = controller_info
        logger.info(f"[{self.client_id}]: Registered Controller server {controller_info.id} at {controller_info.ip}")

    def initialize_storage_manager(
        self,
        manager_type: str,
        config: dict[str, Any],
    ):
        """Initialize the storage manager.

        Args:
            manager_type: Type of storage manager to create. Supported types include:
                          AsyncSimpleStorageManager, KVStorageManager (under development), etc.
            config: Configuration dictionary for the storage manager.
                    For AsyncSimpleStorageManager, must contain the following required keys:
                    - controller_info: ZMQ server information about the controller
                    - storage_unit_infos: ZMQ server information about the storage units

        """
        self.storage_manager = TransferQueueStorageManagerFactory.create(manager_type, config)

    # TODO (TQStorage): Provide a general dynamic socket function for both Client & Storage @huazhong.
    @staticmethod
    def dynamic_socket(socket_name: str):
        """Decorator to auto-manage ZMQ sockets for Controller/Storage servers.

        Handles socket lifecycle: create -> connect -> inject -> close.

        Args:
            socket_name: Port name from server config to use for ZMQ connection (e.g., "data_req_port")

        Decorated Function Requirements:
            1. Must be an async class method (needs `self`)
            2. `self` must have:
               - `_controller`: Server registry
               - `client_id`: Unique client ID for socket identity
            3. Receives ZMQ socket via `socket` keyword argument (injected by decorator)
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                server_info = self._controller
                if not server_info:
                    raise RuntimeError("No controller registered")

                context = zmq.asyncio.Context()
                address = f"tcp://{server_info.ip}:{server_info.ports.get(socket_name)}"
                identity = f"{self.client_id}_to_{server_info.id}_{uuid4().hex[:8]}".encode()
                sock = create_zmq_socket(context, zmq.DEALER, identity=identity)

                try:
                    sock.connect(address)
                    logger.info(
                        f"[{self.client_id}]: Connected to Controller {server_info.id} at {address} "
                        f"with identity {identity.decode()}"
                    )

                    kwargs["socket"] = sock
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    logger.error(f"[{self.client_id}]: Error in socket operation with Controller {server_info.id}: {e}")
                    raise
                finally:
                    try:
                        if not sock.closed:
                            sock.close(linger=-1)
                    except Exception as e:
                        logger.warning(f"[{self.client_id}]: Error closing socket to Controller {server_info.id}: {e}")

                    context.term()

            return wrapper

        return decorator

    @dynamic_socket(socket_name="request_handle_socket")
    async def async_get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        partition_id: str,
        mode: str = "fetch",
        task_name: Optional[str] = None,
        sampling_config: Optional[dict[str, Any]] = None,
        socket: Optional[zmq.asyncio.Socket] = None,
    ) -> BatchMeta:
        """Asynchronously fetch data metadata from the controller via ZMQ.

        Args:
            data_fields: List of data field names to retrieve metadata for
            batch_size: Number of samples to request in the batch
            partition_id: Current data partition id
            mode: Data fetch mode. Options:
                - 'fetch': Get ready data only
                - 'force_fetch': Get data regardless of readiness (may return unready samples)
                - 'insert': Internal usage - should not be used by users
            task_name: Optional task name associated with the request
            sampling_config: Optional sampling configuration for custom samplers.
                           For GRPOGroupNSampler, should include "n_samples_per_prompt": int
            socket: ZMQ async socket for message transmission (injected by decorator)

        Returns:
            BatchMeta: Metadata object containing data structure, sample information, and readiness status

        Raises:
            RuntimeError: If communication fails or controller returns error response

        Example:
            >>> # Example 1: Basic fetch metadata
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["input_ids", "attention_mask"],
            ...     batch_size=4,
            ...     partition_id="train_0",
            ...     mode="fetch",
            ...     task_name="generate_sequences"
            ... ))
            >>> print(batch_meta.is_ready)  # True if all samples ready
            >>>
            >>> # Example 2: Fetch with self-defined samplers (using GRPOGroupNSampler as an example)
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["input_ids", "attention_mask"],
            ...     batch_size=8,
            ...     partition_id="train_0",
            ...     mode="fetch",
            ...     task_name="generate_sequences",
            ...     sampling_config={"n_samples_per_prompt": 4}
            ... ))
            >>> print(batch_meta.is_ready)  # True if all samples ready
            >>>
            >>> # Example 3: Force fetch metadata (bypass production status check and Sampler,
            >>> so may include unready samples. Consumed samples will not be fetched.)
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["input_ids", "attention_mask"],
            ...     batch_size=4,
            ...     partition_id="train_0",
            ...     mode="force_fetch",
            ...     task_name="generate_sequences"
            ... ))
            >>> print(batch_meta.is_ready)  # May be False if some samples not ready
        """
        assert socket is not None
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_META,
            sender_id=self.client_id,
            receiver_id=self._controller.id,
            body={
                "data_fields": data_fields,
                "batch_size": batch_size,
                "partition_id": partition_id,
                "mode": mode,
                "task_name": task_name,
                "sampling_config": sampling_config,
            },
        )

        try:
            await socket.send(request_msg.serialize())
            response = await socket.recv()
            response_msg = ZMQMessage.deserialize(response)
            logger.debug(
                f"[{self.client_id}]: Client get datameta response: {response_msg} "
                f"from controller {self._controller.id}"
            )

            if response_msg.request_type == ZMQRequestType.GET_META_RESPONSE:
                metadata = response_msg.body["metadata"]
                return metadata
            else:
                raise RuntimeError(
                    f"[{self.client_id}]: Failed to get metadata from controller {self._controller.id}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in get_meta: {str(e)}") from e

    async def async_put(
        self,
        data: TensorDict,
        metadata: Optional[BatchMeta] = None,
        partition_id: Optional[str] = None,
    ):
        """Asynchronously write data to storage units based on metadata.

        If metadata is not provided, it will be created automatically using insert mode
        with the provided data fields and partition_id.

        Note:
            When using multiple workers for distributed execution, there may be data
            ordering inconsistencies between workers during put operations.

        Args:
            data: Data to write as TensorDict
            metadata: Records the metadata of a batch of data samples, containing index and
                      storage unit information. If None, metadata will be auto-generated.
            partition_id: Target data partition id (required if metadata is not provided)

        Raises:
            ValueError: If metadata is None or empty, or if partition_id is None when metadata is not provided
            RuntimeError: If storage operation fails

        Example:
            >>> batch_size = 4
            >>> seq_len = 16
            >>> current_partition_id = "train_0"
            >>> # Example 1: Normal usage with existing metadata
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["prompts", "attention_mask"],
            ...     batch_size=batch_size,
            ...     partition_id=current_partition_id,
            ...     mode="fetch",
            ...     task_name="generate_sequences",
            ... ))
            >>> batch = asyncio.run(client.async_get_data(batch_meta))
            >>> output = TensorDict({"response": torch.randn(batch_size, seq_len)})
            >>> asyncio.run(client.async_put(data=output, metadata=batch_meta))
            >>>
            >>> # Example 2: Initial data insertion without pre-existing metadata
            >>> # BE CAREFUL: this usage may overwrite any unconsumed data in the given partition_id!
            >>> # Please make sure the corresponding partition_id is empty before calling the async_put()
            >>> # without metadata.
            >>> # Now we only support put all the data of the corresponding partition id in once. You should repeat with
            >>> # interleave the initial data if n_sample > 1 before calling the async_put().
            >>> original_prompts = torch.randn(batch_size, seq_len)
            >>> n_samples = 4
            >>> prompts_repeated = torch.repeat_interleave(original_prompts, n_samples, dim=0)
            >>> prompts_repeated_batch = TensorDict({"prompts": prompts_repeated})
            >>> # This will create metadata in "insert" mode internally.
            >>> asyncio.run(client.async_put(data=prompts_repeated_batch, partition_id=current_partition_id))

        """

        if not hasattr(self, "storage_manager") or self.storage_manager is None:
            raise RuntimeError(
                f"[{self.client_id}]: Storage manager not initialized. "
                "Call initialize_storage_manager() before performing storage operations."
            )

        if metadata is None:
            if partition_id is None:
                raise ValueError("partition_id must be provided if metadata is not given")

            metadata = await self.async_get_meta(
                data_fields=list(data.keys()),
                batch_size=data.batch_size[0],
                partition_id=partition_id,
                mode="insert",
            )

        if not metadata or metadata.size == 0:
            raise ValueError("metadata cannot be none or empty")
        logger.debug(f"[{self.client_id}]: Put data with data: {data}")

        await self.storage_manager.put_data(data, metadata)

        logger.info(
            f"[{self.client_id}]: partition {partition_id} put {metadata.size} samples to storage units successfully."
        )

    async def async_get_data(self, metadata: BatchMeta) -> TensorDict:
        """Asynchronously fetch data from storage units and organize into TensorDict.

        Args:
            metadata: Batch metadata containing data location information and global indexes

        Returns:
            TensorDict containing:
                - Requested data fields (e.g., "prompts", "attention_mask")

        Example:
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["prompts", "attention_mask"],
            ...     batch_size=4,
            ...     partition_id="train_0",
            ...     mode="fetch",
            ...     task_name="generate_sequences",
            ... ))
            >>> batch = asyncio.run(client.async_get_data(batch_meta))
            >>> print(batch)
            >>> # TensorDict with fields "prompts", "attention_mask", and sample order matching metadata global_indexes

        """

        if not hasattr(self, "storage_manager") or self.storage_manager is None:
            raise RuntimeError(
                f"[{self.client_id}]: Storage manager not initialized. "
                "Call initialize_storage_manager() before performing storage operations."
            )

        if not metadata or metadata.size == 0:
            return TensorDict({}, batch_size=0)

        results = await self.storage_manager.get_data(metadata)

        return results

    async def async_clear(self, partition_id: str):
        """Asynchronously clear data from all storage units and controller metadata.

        Args:
            partition_id: The partition id to clear data for

        Raises:
            RuntimeError: If clear operation fails
        """
        try:
            if not hasattr(self, "storage_manager") or self.storage_manager is None:
                raise RuntimeError(
                    f"[{self.client_id}]: Storage manager not initialized. "
                    "Call initialize_storage_manager() before performing storage operations."
                )

            if not self._controller:
                raise RuntimeError("No controller registered")

            metadata = await self._get_clear_meta(partition_id)

            # Clear the controller metadata
            await self._clear_controller(partition_id)

            # Clear storage unit data
            await self.storage_manager.clear_data(metadata)

            logger.info(f"[{self.client_id}]: Clear operation for partition_id {partition_id} completed.")
        except Exception as e:
            raise RuntimeError(f"Error in clear operation: {str(e)}") from e

    @dynamic_socket(socket_name="request_handle_socket")
    async def _get_clear_meta(self, partition_id: str, socket=None) -> BatchMeta:
        """Get metadata required for clear operation from controller.

        Args:
            partition_id: Partition id to get clear metadata for
            socket: ZMQ socket (injected by decorator)

        Returns:
            BatchMeta: Records the metadata of a batch of data samples.

        Raises:
            RuntimeError: If controller returns error response
        """
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_CLEAR_META,
            sender_id=self.client_id,
            receiver_id=self._controller.id,
            body={"partition_id": partition_id},
        )

        await socket.send(request_msg.serialize())
        serialized = await socket.recv()
        response_msg = ZMQMessage.deserialize(serialized)

        if response_msg.request_type != ZMQRequestType.GET_CLEAR_META_RESPONSE:
            raise RuntimeError(
                f"Failed to get metadata for clear operation: {response_msg.body.get('message', 'Unknown error')}"
            )

        return response_msg.body["metadata"]

    @dynamic_socket(socket_name="request_handle_socket")
    async def _clear_controller(self, partition_id, socket=None):
        """Clear metadata from controller.

        Args:
            partition_id: Partition id to clear metadata for
            socket: ZMQ socket (injected by decorator)

        Raises:
            RuntimeError: If clear operation fails
        """
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_META,
                sender_id=self.client_id,
                receiver_id=self._controller.id,
                body={"partition_id": partition_id},
            )

            await socket.send(request_msg.serialize())
            serialized_msg = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_msg)

            if response_msg.request_type != ZMQRequestType.CLEAR_META_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear controller {self._controller.id}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(
                f"[{self.client_id}]: Successfully clear controller {self._controller.id} for partition_id "
                f"{partition_id}"
            )
        except Exception as e:
            logger.error(f"[{self.client_id}]: Error clearing controller {self._controller.id}: {str(e)}")
            raise

    @dynamic_socket(socket_name="request_handle_socket")
    async def check_data_consumption_status(self, task_name: str, partition_id: str):
        """Check if all samples for current step have been consumed.

        Args:
            task_name: Name of the task to check consumption for
            partition_id: Partition id to check consumption status for
        """
        # TODO: Implement this method to check if all samples for the current step has been consumed
        pass

    @dynamic_socket(socket_name="request_handle_socket")
    async def check_data_production_status(self, data_fields: list[str], partition_id: str):
        """Check if all samples for current partition are ready for consumption.

        Args:
            data_fields: Data fields to check production status for
            partition_id: Partition id to check production status for
        """
        # TODO: Implement this method to check if all samples for the current step is ready for consumption
        pass

    def close(self) -> None:
        """Close the client and cleanup resources including storage manager."""
        try:
            if hasattr(self, "storage_manager") and self.storage_manager:
                if hasattr(self.storage_manager, "close"):
                    self.storage_manager.close()
        except Exception as e:
            logger.warning(f"Error closing storage manager: {e}")


class TransferQueueClient(AsyncTransferQueueClient):
    """Synchronous client wrapper for TransferQueue.

    Provides synchronous versions of all async methods for convenience.
    """

    def __init__(
        self,
        client_id: str,
        controller_info: ZMQServerInfo,
    ):
        """Initialize the synchronous TransferQueue client.

        Args:
            client_id: Unique identifier for this client instance
            controller_info: Single controller ZMQ server information
        """
        super().__init__(
            client_id,
            controller_info,
        )

    def put(self, data: TensorDict, metadata: Optional[BatchMeta] = None, partition_id: Optional[str] = None):
        """Synchronously write data to storage units.

        Args:
            data: Data to write as TensorDict
            metadata: Optional metadata containing index and storage unit information
            partition_id: Target data partition id (required if metadata is not provided)
        """
        return asyncio.run(self.async_put(data, metadata, partition_id))

    def get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        partition_id: str,
        task_name: Optional[str] = None,
        sampling_config: Optional[dict[str, Any]] = None,
    ) -> BatchMeta:
        """Synchronously fetch data metadata from controller.

        Args:
            data_fields: List of data field names to retrieve metadata for
            batch_size: Number of samples to request in the batch
            partition_id: Target data partition id
            task_name: Optional task name associated with the request
            sampling_config: Optional sampling configuration for custom samplers.
                           For GRPOGroupNSampler, should include "n_samples_per_prompt": int

        Returns:
            BatchMeta: Batch metadata containing data location information
        """
        return asyncio.run(
            self.async_get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                partition_id=partition_id,
                task_name=task_name,
                sampling_config=sampling_config,
            )
        )

    def get_data(self, metadata: BatchMeta) -> TensorDict:
        """Synchronously fetch data from storage units.

        Args:
            metadata: Batch metadata containing data location information

        Returns:
            TensorDict containing requested data fields
        """
        return asyncio.run(self.async_get_data(metadata))

    def clear(self, partition_id: str):
        """Synchronously clear data from storage units and controller metadata.

        Args:
            partition_id: The partition id to clear data for
        """
        return asyncio.run(self.async_clear(partition_id))


def process_zmq_server_info(
    handlers: dict[Any, Union["TransferQueueController", "TransferQueueStorageManager", "SimpleStorageUnit"]]
    | Union["TransferQueueController", "TransferQueueStorageManager", "SimpleStorageUnit"],
):  # noqa: UP007
    """Extract ZMQ server information from handler objects.

    Args:
        handlers: Dictionary of handler objects (controllers, storage managers, or storage units),
                  or a single handler object

    Returns:
        If handlers is a dictionary: Dictionary mapping handler names to their ZMQ server information
        If handlers is a single object: ZMQ server information for that object

    Examples:
        >>> # Single handler
        >>> controller = TransferQueueController.remote(...)
        >>> info = process_zmq_server_info(controller)
        >>>
        >>> # Multiple handlers
        >>> handlers = {"storage_0": storage_0, "storage_1": storage_1}
        >>> info_dict = process_zmq_server_info(handlers)"""
    # Handle single handler object case
    if not isinstance(handlers, dict):
        return ray.get(handlers.get_zmq_server_info.remote())  # type: ignore[attr-defined]
    else:
        # Handle dictionary case
        server_info = {}
        for name, handler in handlers.items():
            server_info[name] = ray.get(handler.get_zmq_server_info.remote())  # type: ignore[attr-defined]
        return server_info
