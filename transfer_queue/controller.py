import logging
import math
import os
import threading
import time
from threading import Thread
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
import torch
import zmq
from ray.util import get_node_ip_address

from transfer_queue.metadata import (
    BatchMeta,
    FieldMeta,
    SampleMeta,
)
from transfer_queue.utils.utils import (
    ProductionStatus,
    TransferQueueRole,
    random_sampler,
)
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

TQ_CONTROLLER_GET_METADATA_TIMEOUT = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_TIMEOUT", 300))
TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL", 1))
TQ_INIT_FIELD_NUM = os.environ.get("TQ_INIT_FIELD_NUM", 10)


@ray.remote(num_cpus=1)
class TransferQueueController:
    def __init__(
        self,
        num_storage_units: int,
        global_batch_size: int,
        num_global_batch: int = 1,
        num_n_samples: int = 1,
    ) -> None:
        self.controller_id = f"TQ_CONTROLLER_{uuid4()}"

        self._init_zmq_socket()  # 通过ZMQ实现数据通信

        self.num_storage_units = num_storage_units
        self.global_batch_size = global_batch_size  # 用作global index的offset，区分是哪个global step对应的数据
        self.num_global_batch = num_global_batch
        self.num_n_samples = num_n_samples
        self.total_storage_size = self.global_batch_size * self.num_global_batch * self.num_n_samples

        self.data_production_status = torch.zeros(
            self.total_storage_size, TQ_INIT_FIELD_NUM, dtype=torch.int8
        )  # 默认初始化20个字段，可动态扩展
        # task_name -> consumption_status
        self.data_consumption_status: dict[str, torch.Tensor] = {}
        self.field_name_mapping: dict[str, int] = {}  # 一个data_field和data_status列index的映射表
        # Per-sample dtype and shape storage: {global_index: {field_name: {'dtype': dtype, 'shape': shape}}}
        self.per_tensor_dtype_mapping: dict[int, dict[str, torch.dtype]] = {}
        self.per_tensor_shape_mapping: dict[int, dict[str, torch.Size]] = {}

        self._build_index_storage_mapping()

        self._start_process_handshake()
        self._start_process_update_data_status()
        self._start_process_request()

    def _get_consumer_status(self, task_name: str) -> torch.Tensor:
        # 获取或创建指定消费者的消费状态张量
        if task_name not in self.data_consumption_status:
            # 为新消费者初始化状态
            self.data_consumption_status[task_name] = torch.zeros(self.total_storage_size, dtype=torch.int8)
        return self.data_consumption_status[task_name]

    def _get_per_tensor_dtype(self, global_index: int, field_name: str) -> Optional[torch.dtype]:
        """Get dtype for a specific sample and field."""
        return self.per_tensor_dtype_mapping.get(global_index, {}).get(field_name)

    def _get_per_tensor_shape(self, global_index: int, field_name: str) -> Optional[torch.Size]:
        """Get shape for a specific sample and field."""
        return self.per_tensor_shape_mapping.get(global_index, {}).get(field_name)

    def _step_to_global_index_range(self, global_step: int) -> tuple[int, int]:
        start_idx = (global_step % self.num_global_batch) * self.global_batch_size * self.num_n_samples
        end_idx = start_idx + self.global_batch_size * self.num_n_samples

        return start_idx, end_idx

    def generate_data_status_mask(
        self, data_fields: list[str], global_step: int, task_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 该函数在_get_meta中被调用，根据用户指定的字段和当前的step，生成一个mask矩阵
        # 其中用户指定的字段为入参，当前step对应的行（即global index范围）按照顺序映射即可
        # 该mask矩阵将self.data_production_status中，用户需要的行列选中，
        # 同时将self.data_consumption_status反选，从而生成一个子矩阵，
        # 以便在_get_meta的过程中支持自动向量化操作加速状态查询（直接按行sum判断是否等于shape[1]即可）

        # 检查所有请求的字段是否已注册
        for col in data_fields:
            if col not in self.field_name_mapping:
                # 如果有未注册的列，返回空掩码表示没有可用数据
                empty_row_mask = torch.zeros(self.data_production_status.shape[0], dtype=torch.bool)
                empty_col_mask = torch.zeros(self.data_production_status.shape[1], dtype=torch.bool)
                return empty_row_mask, empty_col_mask

        # step映射到global index
        start_idx, end_idx = self._step_to_global_index_range(global_step)
        row_mask = torch.zeros(self.data_production_status.shape[0], dtype=torch.bool)
        row_mask[start_idx:end_idx] = True

        # 按消费状态反选
        consumer_status = self._get_consumer_status(task_name)
        unconsumed_mask = consumer_status == 0
        row_mask &= unconsumed_mask

        # 选中指定的字段
        col_mask = torch.zeros(self.data_production_status.shape[1], dtype=torch.bool)
        valid_fields = [self.field_name_mapping[col] for col in data_fields]
        if valid_fields:
            col_mask[valid_fields] = True

        return row_mask, col_mask

    def _build_index_storage_mapping(self):
        # 根据数据系统总空间与StorageUnit数量，划分每个Sample应该存储的位置，
        # 并维护global index和每个存储内local index的映射

        # 为每条样本分配存储节点；注意我们应该将每个GBS数据打散在不同存储节点上。
        # 这里和generate_data_status_mask一样，默认按照顺序排列样本
        real_global_batch_size = self.global_batch_size * self.num_n_samples
        global_batch_per_storage_unit = math.ceil(real_global_batch_size / self.num_storage_units)

        # 构建global index与storage unit之间的映射，用于查找每条数据对应的存储节点位置
        batch_storage_indices = np.repeat(np.arange(self.num_storage_units), global_batch_per_storage_unit)[
            :real_global_batch_size
        ]
        self._global_index_storage_rank_mapping = np.tile(batch_storage_indices, self.num_global_batch)

        # 构建global index与每个storage unit之间local index之间的映射
        indices = np.arange(self.total_storage_size)
        pos_in_batch = indices % real_global_batch_size
        g = indices // real_global_batch_size
        pos_in_block = pos_in_batch % global_batch_per_storage_unit
        self.global_index_local_index_mapping = g * global_batch_per_storage_unit + pos_in_block

    def get_data_production_status(self) -> torch.Tensor:
        return self.data_production_status

    def get_field_name_mapping(self) -> dict[str, Any]:
        return self.field_name_mapping

    def get_data_consumption_status(self) -> dict[str, torch.Tensor]:
        return self.data_consumption_status

    def get_global_index_mapping(self):
        return self._global_index_storage_rank_mapping, self.global_index_local_index_mapping

    def _get_metadata(
        self,
        data_fields: list[str],
        batch_size: int,
        mode: str = "fetch",
        global_step=0,
        task_name: str | None = None,
        get_n_samples=False,
        *args,
        **kwargs,
    ) -> BatchMeta:
        """
        获取元数据，支持两种模式：
        - mode="insert": 插入新行的元数据（不检查数据状态）
        - mode="fetch": 获取已就绪的元数据（检查数据状态并采样）
        - mode="force_fetch": 直接返回元数据（不检查数据状态）
        """
        if mode == "insert":
            # TODO 当前仅支持put整个gbs的数据，后续待扩展支持多次put到同一step
            assert batch_size == self.global_batch_size
            start_idx, end_idx = self._step_to_global_index_range(global_step)
            batch_global_indexes = list(range(start_idx, end_idx))
            return self._generate_batch_meta(global_step, batch_global_indexes, data_fields, mode)

        assert task_name is not None
        if mode == "fetch":
            # 向TransferQueue读数据时，查找当前batch内可被消费的样本，并打包返回BatchMeta

            # 循环检查可被消费的数据
            start_time = time.time()
            while True:
                ready_for_consume_idx = self._scan_data_status(data_fields, global_step, task_name, get_n_samples)

                if len(ready_for_consume_idx) >= batch_size:
                    break

                if time.time() - start_time > TQ_CONTROLLER_GET_METADATA_TIMEOUT:
                    raise TimeoutError(
                        f"Timeout while waiting for sufficient data. "
                        f"Required: {batch_size}, Available: {len(ready_for_consume_idx)}"
                    )

                logger.warning(
                    f"Insufficient data available. Required: {batch_size}, "
                    f"Available: {len(ready_for_consume_idx)}. Retrying in {TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                )
                time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
            logger.debug(f"ready for consume idx: {ready_for_consume_idx}")

            batch_global_indexes = random_sampler(ready_for_consume_idx, batch_size, get_n_samples, self.num_n_samples)
        elif mode == "force_fetch":
            start_idx, end_idx = self._step_to_global_index_range(global_step)
            consumer_status = self._get_consumer_status(task_name)
            not_consumed_idx = [i for i in range(start_idx, end_idx) if consumer_status[i] == 0]
            batch_global_indexes = random_sampler(not_consumed_idx, batch_size, get_n_samples, self.num_n_samples)

        # 标记这批数据状态为已消费
        consumer_status = self._get_consumer_status(task_name)
        consumer_status[batch_global_indexes] = 1
        # 打包为metadata
        metadata = self._generate_batch_meta(global_step, batch_global_indexes, data_fields, mode)
        # 6. 如果是方式2，则将metadata进行缓存
        # if dp_rank and dp_size and rank_id:
        #     pass
        logger.debug(f"_get_metadata: {metadata}")

        return metadata

    def _scan_data_status(
        self, data_fields: list[str], global_step: int, task_name: str, get_n_samples: bool
    ) -> list[int]:
        # 获取行和列掩码
        row_mask, col_mask = self.generate_data_status_mask(data_fields, global_step, task_name)
        logger.debug(f"row_mask, col_mask: {row_mask, col_mask}")

        if not row_mask.any() or not col_mask.any():
            return []

        # 提取关注的数据状态子集
        logger.debug(f"self.data_production_status: {self.data_production_status}")
        data_status_of_interest = self.data_production_status[:, col_mask]
        logger.debug(f"data_status_of_interest: {data_status_of_interest}")

        # 使用torch.all向量化检查替代求和比较
        all_fields_ready = torch.all(data_status_of_interest, dim=1)

        # 结合行掩码筛选符合条件的样本
        ready_mask = all_fields_ready & row_mask

        if get_n_samples and self.num_n_samples > 1:
            # 重塑为组视图并检查组完整性
            group_all_ready = torch.all(ready_mask.view(-1, self.num_n_samples), dim=1)

            # 获取完整就绪的组索引
            ready_group_indices = group_all_ready.nonzero(as_tuple=False).flatten()

            # 计算所有样本索引
            sample_offset = torch.arange(self.num_n_samples)
            ready_for_consume_idx = (
                (ready_group_indices.unsqueeze(1) * self.num_n_samples + sample_offset).flatten().tolist()
            )

            return ready_for_consume_idx
        else:
            ready_for_consume_idx = torch.nonzero(ready_mask, as_tuple=False).flatten().tolist()
            logger.debug(f"ready_for_consume_idx: {ready_for_consume_idx}")

            return ready_for_consume_idx

    def _generate_batch_meta(
        self, global_step: int, global_indexes: list[int], data_fields: list[str], mode: str
    ) -> BatchMeta:
        # 根据给定的global index，查找self.global_index_local_index_mapping
        # 和self._global_index_storage_id_mapping，确定对应存储节点的地址，并构建BatchMeta
        global_arr = np.array(global_indexes)
        storage_ids = self.global_index_storage_id_mapping[global_arr]
        local_indexes = self.global_index_local_index_mapping[global_arr]

        samples = []

        # Create samples from the flattened BatchMeta data
        # TODO: 待优化
        for i, global_index in enumerate(global_indexes):
            local_index = local_indexes[i]
            storage_id = storage_ids[i]

            # Create FieldMeta objects for each field
            fields = []
            for field_name in data_fields:
                if mode == "fetch":
                    production_status = ProductionStatus.READY_FOR_CONSUME  # Since we filtered by ready status
                    # Get per-tensor dtype and shape for this specific global_index and field
                    dtype = self._get_per_tensor_dtype(global_index, field_name)
                    shape = self._get_per_tensor_shape(global_index, field_name)
                elif mode == "insert":
                    production_status = ProductionStatus.NOT_PRODUCED  # FIXME: not real-time
                    dtype = None
                    shape = None
                elif mode == "force_fetch":
                    col_index = self.field_name_mapping.get(field_name)
                    if col_index is not None and self.data_production_status[global_index, col_index] == 1:
                        production_status = ProductionStatus.READY_FOR_CONSUME
                        dtype = self._get_per_tensor_dtype(global_index, field_name)
                        shape = self._get_per_tensor_shape(global_index, field_name)
                    else:
                        production_status = ProductionStatus.NOT_PRODUCED
                        dtype = None
                        shape = None
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=dtype,
                    shape=shape,
                    production_status=production_status,
                )
                fields.append(field_meta)

            sample = SampleMeta(
                global_step=global_step,
                global_index=global_index,
                storage_id=storage_id,
                local_index=local_index,
                fields={field.name: field for field in fields},
            )
            samples.append(sample)

        return BatchMeta(samples=samples)

    def _update_production_status(self, indexes: list[int], fields: list[str]) -> None:
        # TODO replace the self.data_production_status == 0 or ==1 operation by using ProductionStatus
        # 更新数据生产状态矩阵
        new_fields = [field for field in fields if field not in self.field_name_mapping]
        if new_fields:
            needed_fields = len(new_fields)
            current_fields = self.data_production_status.shape[1]
            # 扩容数据状态矩阵
            if len(self.field_name_mapping) + needed_fields > current_fields:
                add_fields = max(TQ_INIT_FIELD_NUM, needed_fields + 1)
                new_matrix = torch.zeros((self.total_storage_size, add_fields), dtype=torch.int8)
                self.data_production_status = torch.cat([self.data_production_status, new_matrix], dim=1)

        for field in fields:
            if field not in self.field_name_mapping.keys():
                self.field_name_mapping[field] = len(self.field_name_mapping)
        self.data_production_status[
            torch.tensor(indexes)[:, None], torch.tensor([self.field_name_mapping.get(field) for field in fields])
        ] = 1

    def _update_field_info(
        self,
        fields: list[str],
        per_tensor_dtypes: dict[int, dict[str, Any]],
        per_tensor_shapes: dict[int, dict[str, Any]],
        global_indexes: list[int],
    ) -> None:
        """
        Store per-tensor dtype and shape information.

        Args:
            fields: List of field names
            per_tensor_dtypes: Dict mapping global_index to field dtypes {global_index: {field: dtype}}
            per_tensor_shapes: Dict mapping global_index to field shapes {global_index: {field: shape}}
            global_indexes: List of global indexes corresponding to the samples
        """
        for global_idx in global_indexes:
            if global_idx not in self.per_tensor_dtype_mapping:
                self.per_tensor_dtype_mapping[global_idx] = {}
            if global_idx not in self.per_tensor_shape_mapping:
                self.per_tensor_shape_mapping[global_idx] = {}

            for field in fields:
                if global_idx in per_tensor_dtypes and field in per_tensor_dtypes[global_idx]:
                    self.per_tensor_dtype_mapping[global_idx][field] = per_tensor_dtypes[global_idx][field]
                if global_idx in per_tensor_shapes and field in per_tensor_shapes[global_idx]:
                    self.per_tensor_shape_mapping[global_idx][field] = per_tensor_shapes[global_idx][field]

    def _init_zmq_socket(self):
        # 建立3个ZMQ服务端口，分别用于 ①注册发现 ② 接收Client的数据读写请求 ③ 接收Storage发送的状态更新信号
        self.zmq_context = zmq.Context()

        self._node_ip = get_node_ip_address()
        self._handshake_socket_port = get_free_port()
        self._request_handle_socket_port = get_free_port()
        self._data_status_update_socket_port = get_free_port()

        self.handshake_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.handshake_socket.bind(f"tcp://{self._node_ip}:{self._handshake_socket_port}")

        self.request_handle_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.request_handle_socket.bind(f"tcp://{self._node_ip}:{self._request_handle_socket_port}")

        self.data_status_update_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.data_status_update_socket.bind(f"tcp://{self._node_ip}:{self._data_status_update_socket_port}")

        self.zmq_server_info = ZMQServerInfo.create(
            role=TransferQueueRole.CONTROLLER,
            id=self.controller_id,
            ip=self._node_ip,
            ports={
                "handshake_socket": self._handshake_socket_port,
                "request_handle_socket": self._request_handle_socket_port,
                "data_status_update_socket": self._data_status_update_socket_port,
            },
        )

    def _wait_connection(self):
        # 等待所有存储实例握手;client无需握手以支持动态扩缩容
        # 参考zmq_communication.py中的实现
        # TODO(zjj): 考虑是否需要重传（假设存在Storage没有收到ACK的情况）
        connected_storage_units = set()
        while len(connected_storage_units) < self.num_storage_units:
            identity, serialized_msg = self.handshake_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)
            if request_msg.request_type == ZMQRequestType.HANDSHAKE:
                connected_storage_units.add(request_msg.sender_id)
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.HANDSHAKE_ACK,
                    sender_id=self.controller_id,
                    body={},
                ).serialize()
                self.handshake_socket.send_multipart([identity, response_msg])
                logger.info("Controller send handshake ack successful!")
        self.global_index_storage_id_mapping = np.array(list(connected_storage_units))[
            self._global_index_storage_rank_mapping
        ]
        self.handshake_done.set()

    def _start_process_handshake(self):
        self.handshake_done = threading.Event()
        self.wait_connection_thread = Thread(
            target=self._wait_connection, name="TransferQueueControllerWaitConnectionThread", daemon=True
        )
        self.wait_connection_thread.start()

    def _start_process_update_data_status(self):
        self.process_update_data_status_thread = Thread(
            target=self._update_data_status, name="TransferQueueControllerProcessUpdateDataStatusThread", daemon=True
        )
        self.process_update_data_status_thread.start()

    def _start_process_request(self):
        self.process_request_thread = Thread(
            target=self._process_request, name="TransferQueueControllerProcessRequestThread", daemon=True
        )
        self.process_request_thread.start()

    def _process_request(self):
        # 包含_get_meta、查询当前iteration是否消费完毕等
        self.handshake_done.wait()
        while True:
            # ROUTER套接字接收多部分消息
            identity, serialized_msg = self.request_handle_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)

            if request_msg.request_type == ZMQRequestType.GET_META:
                params = request_msg.body
                logger.info("Controller prepare get metadata...")
                metadata = self._get_metadata(
                    data_fields=params["data_fields"],
                    batch_size=params["batch_size"],
                    global_step=params["global_step"],
                    mode=params.get("mode", "fetch"),
                    task_name=params.get("task_name", None),
                    get_n_samples=params.get("get_n_samples", False),
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )
            elif request_msg.request_type == ZMQRequestType.GET_CLEAR_META:
                params = request_msg.body
                metadata = self._get_metadata(
                    data_fields=[],
                    batch_size=self.global_batch_size,
                    global_step=params["global_step"],
                    mode="insert",
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_CLEAR_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )
            elif request_msg.request_type == ZMQRequestType.CLEAR_META:
                params = request_msg.body
                self.clear(global_step=params["global_step"])
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CLEAR_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"message": f"Clear operation completed by controller {self.controller_id}"},
                )
            elif request_msg.request_type == ZMQRequestType.CHECK_CONSUMPTION:
                # 消费状态检查
                params = request_msg.body
                global_step = params["global_step"]

                consumer_status = self._get_consumer_status(params["task_name"])
                start_idx, end_idx = self._step_to_global_index_range(global_step)
                batch_status = consumer_status[start_idx:end_idx]
                consumed = torch.all(batch_status == 1).item()

                # 构建响应消息
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={
                        "global_step": global_step,
                        "consumed": consumed,
                    },
                )
            self.request_handle_socket.send_multipart([identity, response_msg.serialize()])
            logger.debug("Controller request_handle_socket send_multipart successful!")

    def _update_data_status(self):
        # 用于接受来自storage的数据状态更新信息
        while True:
            logger.debug("Prepare _update_data_status...")
            identity, serialized_msg = self.data_status_update_socket.recv_multipart()
            logger.debug("Controller recv update_data_status request!")
            request_msg = ZMQMessage.deserialize(serialized_msg)
            logger.debug(f"[{self.controller_id}]: Controller recv update_data_status request_msg: {request_msg}")

            if request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE:
                message_data = request_msg.body

                fields = message_data.get("fields", [])
                global_indexes = message_data.get("global_indexes", [])
                per_tensor_dtypes = message_data.get("dtypes", {})  # Now a dict of lists
                per_tensor_shapes = message_data.get("shapes", {})  # Now a dict of lists
                # 更新数据生产状态
                logger.debug(f"global_indexes, fields: {global_indexes, fields}")
                self._update_production_status(global_indexes, fields)
                self._update_field_info(fields, per_tensor_dtypes, per_tensor_shapes, global_indexes)
                logger.info("Controller update production status successful!")

                # 发送确认响应
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Data update acknowledged from controller {self.controller_id}",
                    },
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])
                logger.info("Controller send DATA_UPDATE_ACK successful!")
            elif request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR:
                # 处理数据更新错误
                error_msg = request_msg.body.get("message", "Unknown error")
                print(f"Data update error from storage: {error_msg}")

                # 发送错误确认响应
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Error notification acknowledged from controller {self.controller_id}",
                    },
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])

    def get_zmq_server_info(self) -> ZMQServerInfo:
        return self.zmq_server_info

    def clear(self, global_step: int):
        # 清空对应global batch的数据，当前仅实现单个gbs

        start_idx, end_idx = self._step_to_global_index_range(global_step)

        self.data_production_status[start_idx:end_idx, :] = 0
        for task_name in self.data_consumption_status:
            self.data_consumption_status[task_name][start_idx:end_idx] = 0
