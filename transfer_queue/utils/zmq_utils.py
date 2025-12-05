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

import itertools
import logging
import os
import pickle
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias
from uuid import uuid4

import psutil
import torch
import zmq

try:
    from torch.distributed.rpc.internal import _internal_rpc_pickler

    HAS_RPC_PICKLER = True
except ImportError:
    HAS_RPC_PICKLER = False

from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder
from transfer_queue.utils.utils import (
    ExplicitEnum,
    TransferQueueRole,
    get_env_bool,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

TQ_ZERO_COPY_SERIALIZATION = get_env_bool("TQ_ZERO_COPY_SERIALIZATION", default=False) and HAS_RPC_PICKLER
_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder(torch.Tensor)

bytestr: TypeAlias = bytes | bytearray | memoryview


class ZMQRequestType(ExplicitEnum):
    # HANDSHAKE
    HANDSHAKE = "HANDSHAKE"  # TransferQueueStorageUnit -> TransferQueueController
    HANDSHAKE_ACK = "HANDSHAKE_ACK"  # TransferQueueController  -> TransferQueueStorageUnit

    # DATA_OPERATION
    GET_DATA = "GET"
    PUT_DATA = "PUT"
    GET_DATA_RESPONSE = "GET_DATA_RESPONSE"
    PUT_DATA_RESPONSE = "PUT_DATA_RESPONSE"
    CLEAR_DATA = "CLEAR_DATA"
    CLEAR_DATA_RESPONSE = "CLEAR_DATA_RESPONSE"

    PUT_GET_OPERATION_ERROR = "PUT_GET_OPERATION_ERROR"
    PUT_GET_ERROR = "PUT_GET_ERROR"
    PUT_ERROR = "PUT_ERROR"
    GET_ERROR = "GET_ERROR"
    CLEAR_DATA_ERROR = "CLEAR_DATA_ERROR"

    # META_OPERATION
    GET_META = "GET_META"
    GET_META_RESPONSE = "GET_META_RESPONSE"
    GET_CLEAR_META = "GET_CLEAR_META"
    GET_CLEAR_META_RESPONSE = "GET_CLEAR_META_RESPONSE"
    CLEAR_META = "CLEAR_META"
    CLEAR_META_RESPONSE = "CLEAR_META_RESPONSE"

    # CHECK_CONSUMPTION
    CHECK_CONSUMPTION = "CHECK_CONSUMPTION"
    CONSUMPTION_RESPONSE = "CONSUMPTION_RESPONSE"

    # NOTIFY_DATA_UPDATE
    NOTIFY_DATA_UPDATE = "NOTIFY_DATA_UPDATE"
    NOTIFY_DATA_UPDATE_ACK = "NOTIFY_DATA_UPDATE_ACK"
    NOTIFY_DATA_UPDATE_ERROR = "NOTIFY_DATA_UPDATE_ERROR"


class ZMQServerInfo:
    def __init__(self, role: TransferQueueRole, id: str, ip: str, ports: dict[str, str]):
        self.role = role
        self.id = id
        self.ip = ip
        self.ports = ports

    def to_addr(self, port_name: str) -> str:
        return f"tcp://{self.ip}:{self.ports[port_name]}"

    def to_dict(self):
        return {
            "role": self.role,
            "id": self.id,
            "ip": self.ip,
            "ports": self.ports,
        }

    def __str__(self) -> str:
        return f"ZMQSocketInfo(role={self.role}, id={self.id}, ip={self.ip}, ports={self.ports})"


@dataclass
class ZMQMessage:
    request_type: ZMQRequestType
    sender_id: str
    receiver_id: str | None
    body: dict[str, Any]
    request_id: str
    timestamp: float

    @classmethod
    def create(
        cls,
        request_type: ZMQRequestType,
        sender_id: str,
        body: dict[str, Any],
        receiver_id: Optional[str] = None,
    ) -> "ZMQMessage":
        return cls(
            request_type=request_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            body=body,
            request_id=str(uuid4().hex[:8]),
            timestamp=time.time(),
        )

    def serialize(
        self,
    ) -> list[bytestr]:
        """
        Serializes the ZMQMessage object.

        Returns:
            list[bytestr]: If TQ_ZERO_COPY_SERIALIZATION is enabled, returns a list where the first element
            is the pickled bytes of the message, followed by the flattened serialized tensor parts as
            [pickled_bytes, <bytes>, |<bytes>, <memoryview>, |<bytes>, <memoryview>|...].
            From the third element, two elements is a group that will be used to restore a tensor.

            If TQ_ZERO_COPY_SERIALIZATION is disabled, returns a single-element list containing only the pickled bytes
            through pickle.
        """
        logger.info(f"Serializing ZMQMessage with TQ_ZERO_COPY_SERIALIZATION={TQ_ZERO_COPY_SERIALIZATION}")
        if TQ_ZERO_COPY_SERIALIZATION:
            pickled_bytes, tensors = _internal_rpc_pickler.serialize(self)

            # Process tensors and collect nested tensor info efficiently
            def process_tensor(tensor):
                if tensor.is_nested and tensor.layout == torch.strided:
                    tensor_list = tensor.unbind()
                    tensor_count = len(tensor_list)
                    serialized_tensors = [_encoder.encode(inner_tensor) for inner_tensor in tensor_list]
                    return tensor_count, serialized_tensors
                else:
                    return 1, [_encoder.encode(tensor)]

            # Use map to process all tensors in parallel-like fashion
            nested_tensor_info_and_serialized_tensors = list(map(process_tensor, tensors))

            # Extract nested_tensor_info and flatten serialized tensors using itertools
            nested_tensor_info = [info for info, _ in nested_tensor_info_and_serialized_tensors]
            double_layer_serialized_tensors: list[list[bytestr]] = list(
                itertools.chain.from_iterable(serialized for _, serialized in nested_tensor_info_and_serialized_tensors)
            )
            serialized_tensors: list[bytestr] = list(itertools.chain.from_iterable(double_layer_serialized_tensors))

            return [pickled_bytes, pickle.dumps(nested_tensor_info), *serialized_tensors]
        else:
            return [pickle.dumps(self)]

    @classmethod
    def deserialize(cls, data: list[bytestr] | bytestr) -> "ZMQMessage":
        """Deserialize a ZMQMessage object from serialized data."""
        logger.info(f"Deserializing ZMQMessage with TQ_ZERO_COPY_SERIALIZATION={TQ_ZERO_COPY_SERIALIZATION}")
        if TQ_ZERO_COPY_SERIALIZATION:
            if isinstance(data, list):
                # contain tensors
                pickled_bytes = data[0]
                nested_tensor_info = pickle.loads(data[1])
                serialized_tensors = data[2:]
                if len(serialized_tensors) % 2 != 0:
                    # Note: data is a list of [pickled_bytes, <bytes>, |<bytes>, <memoryview>,
                    # |<bytes>, <memoryview>|...].
                    # From the third element, two elements is a group that will be used to restore a tensor.

                    raise ValueError(
                        f"When TQ_ZERO_COPY_SERIALIZATION is enabled, input data should "
                        f"be a list containing an even number of elements, but got {len(data)}."
                    )
                # deserializing each single tensor
                single_tensors: list[torch.Tensor] = [
                    _decoder.decode(pair)
                    for pair in zip(serialized_tensors[::2], serialized_tensors[1::2], strict=False)
                ]
            else:
                raise ValueError(
                    f"When TQ_ZERO_COPY_SERIALIZATION is enabled, input data should be a list, but got {type(data)}."
                )

            tensor_nums = sum(nested_tensor_info)
            if tensor_nums != len(single_tensors):
                raise ValueError(f"Expecting {tensor_nums} tensors, but got {len(single_tensors)}.")

            tensors = [None] * len(nested_tensor_info)
            current_idx = 0
            for i, tensor_num in enumerate(nested_tensor_info):
                if tensor_num == 1:
                    tensors[i] = single_tensors[current_idx]
                    current_idx += 1
                else:
                    tensors[i] = torch.nested.as_nested_tensor(single_tensors[current_idx : current_idx + tensor_num])
                    current_idx += tensor_num

            return _internal_rpc_pickler.deserialize(pickled_bytes, tensors)
        else:
            if isinstance(data, bytestr):
                return pickle.loads(data)
            elif isinstance(data, list):
                if len(data) > 1:
                    raise ValueError(
                        f"When TQ_ZERO_COPY_SERIALIZATION is disabled, must have only 1 element in"
                        f" list for deserialization, but got {len(data)}."
                    )
                return pickle.loads(data[0])
            else:
                raise ValueError(
                    f"When TQ_ZERO_COPY_SERIALIZATION is disabled, input data should be a list of bytestr,"
                    f" but got {type(data)}."
                )


def get_free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def create_zmq_socket(
    ctx: zmq.Context,
    socket_type: Any,
    identity: Optional[bytestr] = None,
) -> zmq.Socket:
    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)
    return socket
