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
            config: Configuration dictionary for the storage manager. Must contain the
                    following required keys:
                    - data_system_controller_info: ZMQ server information about the controller
                    - data_system_storage_unit_infos: ZMQ server information about the storage units

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
                            sock.setsockopt(zmq.LINGER, -1)
                            sock.close()
                        sock.close(linger=0)
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
        global_step: int,
        mode: str = "fetch",
        get_n_samples: bool = False,
        task_name: Optional[str] = None,
        socket: Optional[zmq.asyncio.Socket] = None,
    ) -> BatchMeta:
        """Asynchronously fetch data metadata from the controller via ZMQ.

        Args:
            data_fields: List of data field names to retrieve metadata for
            batch_size: Number of samples to request in the batch
            global_step: Current training/processing step
            mode: Data fetch mode. Options:
                - 'fetch': Get ready data only
                - 'force_fetch': Get data regardless of readiness (may return unready samples)
                - 'insert': Internal usage - should not be used by users
            get_n_samples: If True, arrange samples of the same prompt contiguously. In 'fetch' mode,
                          only returns samples where all prompts in the group are ready
            task_name: Optional task name associated with the request
            socket: ZMQ async socket for message transmission (injected by decorator)

        Returns:
            BatchMeta: Metadata object containing data structure, sample information, and readiness status

        Raises:
            RuntimeError: If communication fails or controller returns error response

        Example:
            >>> # Example 1: Fetch ready metadata
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["input_ids", "attention_mask"],
            ...     batch_size=4,
            ...     global_step=0,
            ...     mode="fetch",
            ...     task_name="generate_sequences"
            ... ))
            >>> print(batch_meta.is_ready)  # True if all samples ready
            >>>
            >>> # Example 2: Force fetch metadata (may include unready samples)
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["input_ids", "attention_mask"],
            ...     batch_size=4,
            ...     global_step=0,
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
                "global_step": global_step,
                "mode": mode,
                "get_n_samples": get_n_samples,
                "task_name": task_name,
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
        global_step: Optional[int] = None,
    ):
        """Asynchronously write data to storage units based on metadata.

        If metadata is not provided, it will be created automatically using insert mode
        with the provided data fields and global_step.

        Args:
            data: Data to write as TensorDict
            metadata: Records the metadata of a batch of data samples, containing index and
                      storage unit information. If None, metadata will be auto-generated.
            global_step: Current processing step (required if metadata is not provided)

        Raises:
            ValueError: If metadata is None or empty, or if global_step is None when metadata is not provided
            RuntimeError: If storage operation fails

        Example:
            >>> batch_size = 4
            >>> seq_len = 16
            >>> current_step = 0
            >>> # Example 1: Normal usage with existing metadata
            >>> batch_meta = asyncio.run(client.async_get_meta(
            ...     data_fields=["prompts", "attention_mask"],
            ...     batch_size=batch_size,
            ...     global_step=current_step,
            ...     mode="fetch",
            ...     get_n_samples=False,
            ...     task_name="generate_sequences",
            ... ))
            >>> batch = asyncio.run(client.async_get_data(batch_meta))
            >>> output = TensorDict({"response": torch.randn(batch_size, seq_len)})
            >>> asyncio.run(client.async_put(data=output, metadata=batch_meta))
            >>>
            >>> # Example 2: Initial data insertion without pre-existing metadata
            >>> # BE CAREFUL: this usage may overwrite any unconsumed data in the given global_step!
            >>> # Please make sure the corresponding global_step is empty before calling the async_put()
            >>> # without metadata.
            >>> # Now we only support put all the data of the corresponding global step in once. You should repeat with
            >>> # interleave the initial data if n_sample > 1 before calling the async_put().
            >>> original_prompts = torch.randn(batch_size, seq_len)
            >>> n_samples = 4
            >>> prompts_repeated = torch.repeat_interleave(original_prompts, n_samples, dim=0)
            >>> prompts_repeated_batch = TensorDict({"prompts": prompts_repeated})
            >>> # This will create metadata in "insert" mode internally.
            >>> asyncio.run(client.async_put(data=prompts_repeated_batch, global_step=current_step))

        """
        if metadata is None:
            assert global_step is not None, "global_steps must be provided if metadata is not given"

            metadata = await self.async_get_meta(
                data_fields=list(data.keys()),
                batch_size=data.batch_size[0],
                global_step=global_step,
                get_n_samples=True,
                mode="insert",
            )

        if not metadata or metadata.size == 0:
            raise ValueError("metadata cannot be none or empty")
        logger.debug(f"[{self.client_id}]: Put data with data: {data}")

        await self.storage_manager.put_data(data, metadata)

        logger.info(
            f"[{self.client_id}]: step {global_step} put {metadata.size} samples to storage units successfully."
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
            ...     global_step=0,
            ...     mode="fetch",
            ...     get_n_samples=False,
            ...     task_name="generate_sequences",
            ... ))
            >>> batch = asyncio.run(client.async_get_data(batch_meta))
            >>> print(batch)
            >>> # TensorDict with fields "prompts", "attention_mask", and sample order matching metadata global_indexes

        """
        if not metadata or metadata.size == 0:
            return TensorDict({}, batch_size=0)

        results = await self.storage_manager.get_data(metadata)

        return results

    async def async_clear(self, global_step: int):
        """Asynchronously clear data from all storage units and controller metadata.

        Args:
            global_step: The training step to clear data for

        Raises:
            RuntimeError: If clear operation fails
        """
        try:
            if not self._controller:
                raise RuntimeError("No controller registered")

            metadata = await self._get_clear_meta(global_step)

            # Clear the controller metadata
            await self._clear_controller(global_step)

            # Clear storage unit data
            await self.storage_manager.clear_data(metadata)

            logger.info(f"[{self.client_id}]: Clear operation for global_step {global_step} completed.")
        except Exception as e:
            raise RuntimeError(f"Error in clear operation: {str(e)}") from e

    @dynamic_socket(socket_name="request_handle_socket")
    async def _get_clear_meta(self, global_step: int, socket=None) -> BatchMeta:
        """Get metadata required for clear operation from controller.

        Args:
            global_step: Step to get clear metadata for
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
            body={"global_step": global_step},
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
    async def _clear_controller(self, global_step, socket=None):
        """Clear metadata from controller.

        Args:
            global_step: Step to clear metadata for
            socket: ZMQ socket (injected by decorator)

        Raises:
            RuntimeError: If clear operation fails
        """
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_META,
                sender_id=self.client_id,
                receiver_id=self._controller.id,
                body={"global_step": global_step},
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
                f"[{self.client_id}]: Successfully clear controller {self._controller.id} for global_step {global_step}"
            )
        except Exception as e:
            logger.error(f"[{self.client_id}]: Error clearing controller {self._controller.id}: {str(e)}")
            raise

    @dynamic_socket(socket_name="request_handle_socket")
    async def check_current_step_consumption(self, task_name: str, global_step: int):
        """Check if all samples for current step have been consumed.

        Args:
            task_name: Name of the task to check consumption for
            global_step: Step to check consumption status for
        """
        # TODO: Implement this method to check if all samples for the current step has been consumed
        pass

    @dynamic_socket(socket_name="request_handle_socket")
    async def check_current_step_production(self, data_fields: list[str], global_step: int):
        """Check if all samples for current step are ready for consumption.

        Args:
            data_fields: Data fields to check production status for
            global_step: Step to check production status for
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

    def put(self, data: TensorDict, metadata: Optional[BatchMeta] = None, global_step: Optional[int] = None):
        """Synchronously write data to storage units.

        Args:
            data: Data to write as TensorDict
            metadata: Optional metadata containing index and storage unit information
            global_step: Current processing step (required if metadata is not provided)
        """
        return asyncio.run(self.async_put(data, metadata, global_step))

    def get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        global_step: int,
        get_n_samples: bool = False,
        task_name: Optional[str] = None,
    ) -> BatchMeta:
        """Synchronously fetch data metadata from controller.

        Args:
            data_fields: List of data field names to retrieve metadata for
            batch_size: Number of samples to request in the batch
            global_step: Current training/processing step
            get_n_samples: If True, arrange samples of the same prompt contiguously
            task_name: Optional task name associated with the request

        Returns:
            BatchMeta: Batch metadata containing data location information
        """
        return asyncio.run(
            self.async_get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                global_step=global_step,
                get_n_samples=get_n_samples,
                task_name=task_name,
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

    def clear(self, global_step: int):
        """Synchronously clear data from storage units and controller metadata.

        Args:
            global_step: The training step to clear data for
        """
        return asyncio.run(self.async_clear(global_step))


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
