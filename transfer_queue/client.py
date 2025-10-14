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
from typing import Any, Callable, Optional
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
    TransferQueueStorageManager,
    TransferQueueStorageManagerFactory,
    TransferQueueStorageSimpleUnit,
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
    def __init__(
        self,
        client_id: str,
        controller_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        self.client_id = client_id
        self._controllers: dict[str, ZMQServerInfo] = {}
        self._register_controllers(controller_infos)

    def initialize_storage_manager(
        self,
        manager_type: str,
        config: dict[str, Any],
    ):
        self.storage_manager = TransferQueueStorageManagerFactory.create(manager_type, config)

    def _register_controllers(
        self,
        server_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        mapping = self._controllers

        if not isinstance(server_infos, dict):
            server_infos = {server_infos.id: server_infos}

        for info in server_infos.values():
            if not isinstance(info, ZMQServerInfo):
                raise ValueError(f"Invalid server info, expecting ZMQServerInfo, but get type {type(info)}")

            if info.id not in mapping:
                mapping[info.id] = info
                logger.info(f"[{self.client_id}]: Registered Controller server {info.id} at {info.ip}")
            else:
                logger.warning(f"[{self.client_id}]: Server {info.id} already registered, skipping")

    @staticmethod
    def dynamic_socket(socket_name: str):
        """Decorator to auto-manage ZMQ sockets for Controller/Storage servers (create -> connect -> inject -> close).

        Args:
            socket_name (str): Port name (from server config) to use for ZMQ connection (e.g., "data_req_port").

        Decorated Function Rules:
            1. Must be an async class method (needs `self`).
            2. `self` requires:
            - `_controllers`: Server registries (match `target_role`).
            - `client_id`: Unique client ID (for socket identity).
            4. Receives ZMQ socket via `socket` keyword arg (injected by decorator).
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                server_key = kwargs.get("target_controller")
                if server_key is None:
                    for arg in args:
                        if isinstance(arg, str) and arg in self._controllers.keys():
                            server_key = arg
                            break

                if server_key is None:
                    server_key = next(iter(self._controllers.keys()))

                server_info = self._controllers.get(server_key)
                if not server_info:
                    raise RuntimeError(f"Server {server_key} not found in registered Controller servers")

                context = zmq.asyncio.Context()
                address = f"tcp://{server_info.ip}:{server_info.ports.get(socket_name)}"
                identity = f"{self.client_id}_to_{server_info.id}_{uuid4()}".encode()
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
        target_controller: Optional[str] = None,
        socket: Optional[zmq.asyncio.Socket] = None,
    ) -> BatchMeta:
        """Asynchronously fetches data metadata via ZMQ from the target controller.

        Args:
            data_fields (list[str]): List of fields to retrieve metadata for
            batch_size (int): Processing batch size
            global_step (int): Current training/processing step
            mode (str): Data fetch mode. 'fetch' to get ready data, 'force_fetch' to get data regardless of readiness.
                        'insert' IS AN INTERNAL USAGE THAT SHOULD NOT BE USED BY USERS.
            get_n_samples (bool): If True, we arrange the samples of the same prompt in contiguous order. In 'fetch'
                                  mode, only the samples of the same prompt that are all ready will be returned.
            task_name (str): Optional task name associated with the request
            target_controller (str): ID of the target controller to send the request to
            socket (zmq.asyncio.Socket): ZMQ async socket for message transmission

        Example:
            >>> batch_size = 4
            >>> current_step = 0
            >>> # Example 1: "fetch" a batch of metadata that has been produced
            >>> batch_meta = asyncio.run(client.async_get_meta(data_fields=["input_ids", "attention_mask"],
            >>>                                                batch_size=batch_size,
            >>>                                                global_step=current_step,
            >>>                                                mode="fetch",
            >>>                                                get_n_samples=False,
            >>>                                                task_name="generate_sequences",
            >>>                                                ))
            >>> print(batch_meta.is_ready)   # you should get a batch_meta with is_ready=True
            >>> print([sample_meta.is_ready for sample_meta in batch_meta.samples])  # [True, True, True, True]
            >>>
            >>> # Example 2: "force_fetch" a batch of metadata, ignoring their production status (but we still make
            >>> # sure the corresponding data has not been consumed)
            >>> batch_meta = asyncio.run(client.async_get_meta(data_fields=["input_ids", "attention_mask"],
            >>>                                                batch_size=batch_size,
            >>>                                                global_step=current_step,
            >>>                                                mode="force_fetch",
            >>>                                                get_n_samples=False,
            >>>                                                task_name="generate_sequences",
            >>>                                                ))
            >>> print(batch_meta.is_ready)   # you may get a batch_meta with is_ready=False
            >>> print([sample_meta.is_ready for sample_meta in batch_meta.samples])  # [True, False, False, True]

        Returns:
            BatchMeta: Metadata object containing data structure, sample info, etc.
        """
        assert socket is not None
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_META,
            sender_id=self.client_id,
            receiver_id=target_controller,
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
                f"[{self.client_id}]: Client get datameta response: {response_msg} from controller {target_controller}"
            )

            if response_msg.request_type == ZMQRequestType.GET_META_RESPONSE:
                metadata = response_msg.body["metadata"]
                return metadata
            else:
                raise RuntimeError(
                    f"[{self.client_id}]: Failed to get metadata from controller {target_controller}: "
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
        """Asynchronously writes data to appropriate Storage Units based on metadata.

        If metadata isn't provided, it will be created automatically using the insert mode
        with the provided data_columns and global_step.

        Args:
            data (torch.Tensor | tensordict.TensorDict): Data to write, either a Tensor or TensorDict
            metadata (BatchMeta, optional): Optional metadata containing index and storage unit information
            global_step (int, optional): Current step (required if no metadata is provided)

        Example:
            >>> batch_size = 4
            >>> seq_len = 16
            >>> current_step = 0
            >>> # Example 1: normal usage
            >>> batch_meta = asyncio.run(client.async_get_meta(data_fields=["prompts", "attention_mask"],
            >>>                                   batch_size=batch_size,
            >>>                                   global_step=current_step,
            >>>                                   mode="fetch",
            >>>                                   get_n_samples=False,
            >>>                                   task_name="generate_sequences",
            >>>                                   ))
            >>> batch = asyncio.run(client.async_get_data(batch_meta))
            >>> output = TensorDict({"response": torch.randn(batch_size, seq_len)})
            >>> asyncio.run(client.async_put(data=output, metadata=batch_meta))
            >>>
            >>> # Example 2: put the initial data into the system without pre-existing metadata
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
        """Asynchronously fetches data via Storage Units and organizes it into a TensorDict.

        Args:
            metadata (BatchMeta): Object containing:
                - Data location info (which Storage Units hold the data)
                - `global_indexes` to determine the ordering of merged results

        Returns:
            tensordict.TensorDict with:
                - Requested data fields (e.g., "prompt_token_ids", "response_token_ids").
                - "global_indexes" key: Maps each sample to its original global index.

        Example:
            >>> batch_size = 4
            >>> seq_len = 16
            >>> current_step = 0
            >>> batch_meta = asyncio.run(client.async_get_meta(data_fields=["prompts", "attention_mask"],
            >>>                                   batch_size=batch_size,
            >>>                                   global_step=current_step,
            >>>                                   mode="fetch",
            >>>                                   get_n_samples=False,
            >>>                                   task_name="generate_sequences",
            >>>                                   ))
            >>> batch = asyncio.run(client.async_get_data(batch_meta))
            >>> print(batch)
            >>> # this is a TensorDict with fields "prompts" and "attention_mask".
            >>> # The order of samples in the TensorDict matches the order of global_indexes in batch_meta

        Note:
            Why track `global_indexes`?
            - Batches may be rearranged during task processing. `global_indexes` retains the original
            mapping to Storage Units, enabling correct data writing back to Storage Units later.

        """
        if not metadata or metadata.size == 0:
            return TensorDict({}, batch_size=0)

        results = await self.storage_manager.get_data(metadata)

        return results

    async def async_clear(self, global_step: int):
        """Asynchronously clears data from all storage units and controller metadata.

        Args:
            global_step (int): The training step associated with the clear operation

        """
        try:
            target_controller = next(iter(self._controllers.keys()))
            metadata = await self._get_clear_meta(global_step, target_controller)

            tasks = []

            for target_controller in self._controllers.keys():
                tasks.append(self._clear_controller(global_step, target_controller))

            tasks.append(self.storage_manager.clear_data(metadata))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[{self.client_id}]: Error in clear operation task {i}: {result}")

            logger.info(f"[{self.client_id}]: Clear operation for global_step {global_step} completed.")
        except Exception as e:
            raise RuntimeError(f"Error in clear operation: {str(e)}") from e

    @dynamic_socket(socket_name="request_handle_socket")
    async def _get_clear_meta(self, global_step: int, target_controller=None, socket=None) -> BatchMeta:
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_CLEAR_META,
            sender_id=self.client_id,
            receiver_id=target_controller,
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
    async def _clear_controller(self, global_step, target_controller=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_META,
                sender_id=self.client_id,
                receiver_id=target_controller,
                body={"global_step": global_step},
            )

            await socket.send(request_msg.serialize())
            serialized_msg = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_msg)

            if response_msg.request_type != ZMQRequestType.CLEAR_META_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear controller {target_controller}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(
                f"[{self.client_id}]: Successfully clear controller {target_controller} for global_step {global_step}"
            )
        except Exception as e:
            logger.error(f"[{self.client_id}]: Error clearing controller {target_controller}: {str(e)}")
            raise

    @dynamic_socket(socket_name="request_handle_socket")
    def check_current_step_consumption(self, task_name: str, global_step: int):
        # TODO: Implement this method to check if all samples for the current step has been consumed
        pass

    @dynamic_socket(socket_name="request_handle_socket")
    def check_current_step_production(self, data_fields: list[str], global_step: int):
        # TODO: Implement this method to check if all samples for the current step is ready for consumption
        pass


class TransferQueueClient(AsyncTransferQueueClient):
    def __init__(
        self,
        client_id: str,
        controller_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
        storage_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        super().__init__(
            client_id,
            controller_infos,
            storage_infos,
        )

    def put(self, data: TensorDict, metadata: Optional[BatchMeta] = None, global_step: Optional[int] = None):
        return asyncio.run(self.async_put(data, metadata, global_step))

    def get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        global_step: int,
        get_n_samples: bool = False,
        task_name: Optional[str] = None,
    ) -> BatchMeta:
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
        return asyncio.run(self.async_get_data(metadata))

    def clear(self, global_step: int):
        return asyncio.run(self.async_clear(global_step))


def process_zmq_server_info(
    handlers: dict[Any, TransferQueueController | TransferQueueStorageManager | TransferQueueStorageSimpleUnit],
):  # noqa: UP007
    server_info = {}
    for name, handler in handlers.items():
        server_info[name] = ray.get(handler.get_zmq_server_info.remote())  # type: ignore[attr-defined]
    return server_info
