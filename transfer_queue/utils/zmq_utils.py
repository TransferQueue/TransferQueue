import time
import socket
from typing import Any, Optional, Dict, List, Union
from typing_extensions import Self
import psutil
import uuid
import zmq
import pickle
from dataclasses import dataclass

from transfer_queue.utils.utils import (
    ExplicitEnum,
    TransferQueueRole,
)



class ZMQRequestType(ExplicitEnum):
    # TODO @jianjun: 添加注释，说明每个Request type可能对应的通信过程

    # 握手相关
    HANDSHAKE = "HANDSHAKE"               # TransferQueueStorageUnit -> TransferQueueController
    HANDSHAKE_ACK = "HANDSHAKE_ACK"       # TransferQueueController  -> TransferQueueStorageUnit

    # 数据操作相关
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

    # 元数据相关
    GET_META = "GET_META"
    GET_META_RESPONSE = "GET_META_RESPONSE"
    GET_CLEAR_META = "GET_CLEAR_META"
    GET_CLEAR_META_RESPONSE = "GET_CLEAR_META_RESPONSE"
    CLEAR_META = "CLEAR_META"
    CLEAR_META_RESPONSE = "CLEAR_META_RESPONSE"

    # 消费状态相关
    CHECK_CONSUMPTION = "CHECK_CONSUMPTION"
    CONSUMPTION_RESPONSE = "CONSUMPTION_RESPONSE"

    # 数据更新通知相关
    NOTIFY_DATA_UPDATE = "NOTIFY_DATA_UPDATE"
    NOTIFY_DATA_UPDATE_ACK = "NOTIFY_DATA_UPDATE_ACK"
    NOTIFY_DATA_UPDATE_ERROR = "NOTIFY_DATA_UPDATE_ERROR"


@dataclass
class ZMQServerInfo:
    role: TransferQueueRole
    id: str
    ip: str
    ports: Dict[str, str]

    @classmethod
    def create(cls, role: TransferQueueRole, id: str, ip: str, ports:  Dict[str, str]) -> Self:
        return cls(
            role=role,
            id=id,
            ip=ip,
            ports=ports
        )

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
    receiver_id: Union[str, None]
    body: Dict[str, Any]
    request_id: str
    timestamp: float

    @classmethod
    def create(cls, request_type: ZMQRequestType, sender_id: str, body: Dict[str, Any], receiver_id: Optional[Union[str, None]] = None) -> "ZMQMessage":
        return cls(
            request_type=request_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            body=body,
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )

    def serialize(self) -> bytes:
        """使用pickle序列化ZMQMessage对象"""
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: Union[bytes, List[bytes]]):
        """
        使用pickle反序列化ZMQMessage
        """
        if isinstance(data, list):
            # 处理多个字节流的情况，按顺序反序列化每个字节流
            result = []
            for d in data:
                result = result.append(pickle.loads(d))
            return result
        else:
            # 单个字节流的情况
            return pickle.loads(data)


def get_free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def create_zmq_socket(
    ctx: zmq.Context,
    socket_type: Any,
    identity: Optional[bytes] = None,
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
