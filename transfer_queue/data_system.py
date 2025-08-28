import os
import time
import multiprocessing
from typing import List, Dict, Optional, Union, Any, Callable, TypedDict
from enum import Enum
import ray
import numpy as np
import torch
from torch import Tensor
from tensordict import TensorDict
from abc import ABC
from threading import Thread
import zmq
import math
import logging
from uuid import uuid4
from dataclasses import dataclass, field

from .zmq_utils import ZMQRequestType, ZMQMessage, ZMQServerInfo, get_free_port, get_node_ip, create_zmq_socket, \
    TransferQueueRole
from .load_balance_strategy import random_strategy, dp_token_load_balancing_strategy, similar_seqlen_strategy, \
    storage_unit_load_balancing_strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT = os.environ.get("CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT", 30)
CONTROLLER_DATA_UPDATE_RESPONSE_TIMEOUT = os.environ.get("CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT", 600)
POLLER_MAX_SIZE = os.environ.get("POLLER_MAX_SIZE", 1000)
CONTROLLER_GET_METADATA_TIMEOUT = os.environ.get("CONTROLLER_GET_METADATA_TIMEOUT", 300)
CONTROLLER_GET_METADATA_CHECK_INTERVAL = os.environ.get("CONTROLLER_GET_METADATA_CHECK_INTERVAL", 1)


class IndexData(TypedDict):
    columns: List[str]
    local_indexes: List[int]
    global_indexes: List[int]


@dataclass
class ExperienceMeta:
    """

    global_indexes: 全局索引列表
    columns: 列名列表
    local_indexes: 局部索引列表
    storage_unit_ranks: 存储单元rank列表
    """
    # 使用default_factory来生成每个实例独立的空列表，避免初始化多个实例时可变对象默认值共享的问题
    global_indexes: List[int] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    local_indexes: List[int] = field(default_factory=list)
    # TODO: ID格式统一，用ID str替代rank编号管理数据 @jianjun；包括后面用rank id管理的一些逻辑
    storage_unit_ranks: List[int] = field(default_factory=list)

    # 建立 storage unit rank: Dict(columns: List[str], local_indexes: List[int], global_indexes: List[int])的映射
    storage_unit_index_map: Dict[int, IndexData] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """初始化时构建storage unit - index字典"""
        self._build_storage_unit_index_mapping()

    def _build_storage_unit_index_mapping(self):
        self.storage_unit_index_map.clear()

        # 确保所有列表长度一致
        if not (len(self.global_indexes) == len(self.local_indexes) == len(
                self.storage_unit_ranks)):
            raise ValueError("索引列表的长度必须一致")

        for i, storage_unit_rank in enumerate(self.storage_unit_ranks):
            # 初始化每个storage_unit_rank的字典
            if storage_unit_rank not in self.storage_unit_index_map:
                self.storage_unit_index_map[storage_unit_rank] = {
                    'columns': [],
                    'local_indexes': [],
                    'global_indexes': []
                }

            self.storage_unit_index_map[storage_unit_rank]['columns'] = self.columns
            self.storage_unit_index_map[storage_unit_rank]['local_indexes'].append(self.local_indexes[i])
            self.storage_unit_index_map[storage_unit_rank]['global_indexes'].append(self.global_indexes[i])

    @property
    def size(self) -> int:
        """返回当前块中的数据条目数"""
        return len(self.global_indexes)

    def set_target_column(self, target_column: List[str]):
        self.columns = target_column
        for rank in self.storage_unit_index_map:
            self.storage_unit_index_map[rank]['columns'] = target_column

    def chunk(self, chunk_size: int) -> List['ExperienceMeta']:
        """
        将当前数据块分割成更小的块

        参数:
        chunk_size: 每个小块的大小

        返回:
        分割后的小块列表
        """
        chunks = []
        n = len(self.global_indexes)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = ExperienceMeta(
                global_indexes=self.global_indexes[start:end],
                columns=self.columns.copy(),  # 复制列名列表
                local_indexes=self.local_indexes[start:end],
                storage_unit_ranks=self.storage_unit_ranks[start:end]
            )
            chunks.append(chunk)

        return chunks

    @classmethod
    def concat(
            cls,
            chunks: List['ExperienceMeta'],
            validate: bool = True
    ) -> Optional['ExperienceMeta']:
        """
        连接多个数据块为一个大数据块

        参数:
        chunks: 要连接的数据块列表
        validate: 是否验证连接条件

        返回:
        合并后的数据块 (失败时返回None)
        """
        if not chunks:
            return None

        # 验证连接条件
        if validate:
            base_columns = chunks[0].columns

            for chunk in chunks:
                if chunk.columns != base_columns:
                    print("错误: 列名不一致")
                    return None

        # 合并数据
        all_global_indexes = []
        all_local_indexes = []
        all_storage_list = []

        for chunk in chunks:
            all_global_indexes.extend(chunk.global_indexes)
            all_local_indexes.extend(chunk.local_indexes)
            all_storage_list.extend(chunk.storage_unit_ranks)

        return ExperienceMeta(
            global_indexes=all_global_indexes,
            columns=chunks[0].columns.copy(),
            local_indexes=all_local_indexes,
            storage_unit_ranks=all_storage_list
        )


@ray.remote(max_concurrency=50, num_cpus=4)
class TransferQueueController:
    def __init__(self, controller_id: str, num_storage_units: int, global_batch_size: int, num_global_batch: int = 1,
                 num_n_samples: int = 1, customized_sample_policy: Optional[Dict[str, Callable]] = {},
                 ) -> None:

        # self.controller_id = f"TQ_CONTROLLER_{uuid4()}"
        self.controller_id = controller_id  # FIXME 临时回滚：后续不在主控指定ID @jianjun

        self._init_zmq_socket()  # 通过ZMQ实现数据通信

        self.num_storage_units = num_storage_units
        self.global_batch_size = global_batch_size  # 用作global index的offset，区分是哪个global step对应的数据
        self.num_global_batch = num_global_batch
        self.num_n_samples = num_n_samples
        self.total_storage_size = self.global_batch_size * self.num_global_batch * self.num_n_samples

        self.data_production_status = torch.zeros(self.total_storage_size, 20, dtype=torch.int8)  # 默认初始化20列，可动态扩展
        self.data_consumption_status = {}  # Dict[bytes, torch.Tensor] (task_name -> 消费状态张量)
        self.column_name_mapping = {}  # 一个data_column和data_status列index的映射表
        # 例如：{'Prompt':0, 'Response':1, ...}

        # 用于支持每个rank自行获取数据的场景
        self.dp_metadata_buffer = {}  # 例：{'DP0':ExperienceMeta, 'DP1':ExperienceMeta}
        self.dp_rank_consumption = {}  # 例：{'DP0':set(), 'DP1':set()}  # 其中set记录已经消费过这个metadata的rank_id

        self._build_index_storage_mapping()
        self._register_default_policy()

        # User may pass multiple customized policies for different tasks' sampling
        for customized_policy_name, customized_policy_func in customized_sample_policy.items():
            if customized_policy_name is not None and customized_policy_func is not None:
                self._register_customized_policy(customized_policy_name, customized_policy_func)

        self.wait_connection_thread = Thread(target=self._wait_connection,
                                             name="TransferQueueControllerWaitConnectionThread",
                                             daemon=True)
        self.wait_connection_thread.start()
        self._start_process_request()

    def _get_consumer_status(self, task_name: str) -> torch.Tensor:
        # 获取或创建指定消费者的消费状态张量
        if task_name not in self.data_consumption_status:
            # 为新消费者初始化状态
            self.data_consumption_status[task_name] = torch.zeros(
                self.total_storage_size,
                dtype=torch.int8
            )
        return self.data_consumption_status[task_name]

    def generate_data_status_mask(self, data_columns: List[str], global_step: int, task_name: str) -> (
            Tensor, Tensor):
        # 该函数在_get_meta中被调用，根据用户指定的列和当前的step，生成一个mask矩阵
        # 其中用户指定的列为入参，当前step对应的行（即global index范围）按照顺序映射即可
        # 该mask矩阵将self.data_production_status中，用户需要的行列选中，同时将self.data_consumption_status反选，
        # 从而生成一个子矩阵，以便在_get_meta的过程中支持自动向量化操作加速状态查询（直接按行sum判断是否等于shape[1]即可）

        # step映射到global index
        start_idx = (global_step % self.num_global_batch) * self.global_batch_size * self.num_n_samples
        end_idx = start_idx + self.global_batch_size * self.num_n_samples
        row_mask = torch.zeros(self.data_production_status.shape[0], dtype=torch.bool)
        row_mask[start_idx:end_idx] = True

        # 按消费状态反选
        consumer_status = self._get_consumer_status(task_name)
        unconsumed_mask = (consumer_status == 0)
        row_mask &= unconsumed_mask

        # 选中指定的列
        col_mask = torch.zeros(self.data_production_status.shape[1], dtype=torch.bool)
        valid_columns = [self.column_name_mapping[col] for col in data_columns]
        if valid_columns:
            col_mask[valid_columns] = True

        return row_mask, col_mask

    def _build_index_storage_mapping(self):
        # 根据数据系统总空间与StorageUnit数量，划分每个Sample应该存储的位置，并维护global index和每个存储内local index的映射

        # 为每条样本分配存储节点；注意我们应该将每个GBS数据打散在不同存储节点上。这里和generate_data_status_mask一样，默认按照顺序排列样本
        real_global_batch_size = self.global_batch_size * self.num_n_samples
        global_batch_per_storage_unit = math.ceil(real_global_batch_size / self.num_storage_units)

        # 构建global index与storage unit之间的映射，用于查找每条数据对应的存储节点位置
        batch_storage_indices = np.repeat(np.arange(self.num_storage_units),
                                          global_batch_per_storage_unit)[:real_global_batch_size]
        self.global_index_storage_mapping = np.tile(batch_storage_indices, self.num_global_batch)  # TODO: storage_unit_ranks -> storage_unit_ids, 这里改成字典 @jianjun

        # 构建global index与每个storage unit之间local index之间的映射
        indices = np.arange(self.total_storage_size)
        pos_in_batch = indices % real_global_batch_size
        g = indices // real_global_batch_size
        pos_in_block = pos_in_batch % global_batch_per_storage_unit
        self.global_index_local_index_mapping = g * global_batch_per_storage_unit + pos_in_block

    # DEPRECATED：第一阶段只调通主控拿metadata+worker拿data，因此暂时无需维护dp身份感知的功能，即
    # num_dp_groups: int,
    # dp_rank: int = None,
    # dp_size: int = None,
    # rank_id: int = None,
    # 无需设计
    # def _get_metadata(self,
    #                   data_columns:List[str],
    #                   experience_count:int,
    #                   current_step: int,
    #                   dp_world_size:int,
    #                   num_dp_groups:int=None,
    #                   dp_rank:int=None,
    #                   rank_id:int=None,
    #                   get_n_samples=False,
    #                   schedule_policy:str='DP_balance',
    #                   *args,
    #                   **kwargs) -> ExperienceMeta:
    #     # 向TransferQueue读数据时，查找当前batch内可被消费的样本，并打包返回ExperienceMeta
    #
    #     # 为保证兼容性，当前考虑支持两种使用方式：
    #     # 方式1：主控读取所有DP的metadata，通过dispatch进行分发。此时无需指定dp_rank与dp_size
    #     # 方式2：每个Rank自行请求数据，这时需要指定dp_rank与dp_size，在TransferQueue系统内保证相同DP拿到相同数据、不同DP拿到不同数据
    #
    #     # 1. 根据是否指定dp_rank、dp_size、rank_id，判断是否需要记录请求队列
    #     if dp_rank and dp_size and rank_id:
    #         if dp_rank in self.dp_metadata_buffer.keys():
    #             # 说明该dp_rank中其他的某张卡已经发送过数据读取请求
    #             if rank_id not in self.dp_rank_consumption['DP'+str(dp_rank)]:
    #                 # 说明当前rank没有消费过这个batch的数据，直接从buffer中读取metadata
    #                 metadata = self.dp_metadata_buffer['DP'+str(dp_rank)]
    #                 self.dp_rank_consumption['DP'+str(dp_rank)].add(rank_id)
    #                 if len(self.dp_rank_consumption['DP'+str(dp_rank)]) == dp_size:
    #                     # 这批数据已经被DP域内所有rank消费过，逐出
    #                     del(self.dp_rank_consumption['DP'+str(dp_rank)])
    #                     del(self.dp_metadata_buffer['DP'+str(dp_rank)])
    #                 return metadata
    #             else:
    #                 # 异常处理，DP域内某个rank在其他rank没有计算完的时候又发了一个请求，抛出异常
    #                 pass
    #
    #     # 执行至此，说明需要重新采样一批数据
    #     # 2. 扫描数据状态，找到所有可消费数据
    #     ready_for_consume_idx = self._scan_data_status(data_columns, current_step, get_n_samples)
    #     # 3. 执行负载均衡，采样一批数据
    #     batch_global_indexes = self._run_schedule_policy(schedule_policy, experience_count, ready_for_consume_idx, *args, **kwargs)
    #     # 4. 标记这批数据状态为已消费
    #     self.data_consumption_status[batch_global_indexes] = 1
    #     # 5. 打包为metadata
    #     metadata = self._generate_experience_meta(batch_global_indexes,data_columns)
    #     # 6. 如果是方式2，则将metadata进行缓存
    #     if dp_rank and dp_size and rank_id:
    #         pass
    #
    #     return metadata

    def _get_metadata(self,
                      data_columns: List[str],
                      experience_count: int,
                      dp_world_size: int,
                      task_name: str,
                      get_n_samples=False,
                      global_step=0,
                      schedule_policy: str = 'dp_token_load_balancing_strategy',
                      *args,
                      **kwargs) -> ExperienceMeta:
        # 向TransferQueue读数据时，查找当前batch内可被消费的样本，并打包返回ExperienceMeta
        
        # 循环检查可被消费的数据
        start_time = time.time()
        while True:
            ready_for_consume_idx = self._scan_data_status(data_columns, global_step, task_name, get_n_samples)
            
            if len(ready_for_consume_idx) >= experience_count:
                break
                
            if time.time() - start_time > CONTROLLER_GET_METADATA_TIMEOUT:
                raise TimeoutError(
                    f"Timeout while waiting for sufficient data. "
                    f"Required: {experience_count}, Available: {len(ready_for_consume_idx)}"
                )
            
            logger.warning(
                f"Insufficient data available. Required: {experience_count}, "
                f"Available: {len(ready_for_consume_idx)}. Retrying in {CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
            )
            time.sleep(CONTROLLER_GET_METADATA_CHECK_INTERVAL)
        logger.info(f"ready for consume idx: {ready_for_consume_idx}")
        # 执行负载均衡，采样一批数据
        batch_global_indexes = self._run_schedule_policy(schedule_policy, ready_for_consume_idx, experience_count,
                                                         *args, **kwargs)
        # 标记这批数据状态为已消费
        consumer_status = self._get_consumer_status(task_name)
        consumer_status[batch_global_indexes] = 1
        # 打包为metadata
        metadata = self._generate_experience_meta(batch_global_indexes, data_columns)
        # 6. 如果是方式2，则将metadata进行缓存
        # if dp_rank and dp_size and rank_id:
        #     pass
        logger.info(f"_get_metadata: {metadata}")

        return metadata

    def _get_prompt_metadata(self,
                             data_columns: List[str],
                             experience_count: int,
                             global_step: int,
                             n_samples_per_prompt: int,
                             ) -> ExperienceMeta:
        start_idx = (global_step % self.num_global_batch) * experience_count * n_samples_per_prompt
        end_idx = start_idx + experience_count * n_samples_per_prompt

        batch_global_indexes = list(range(start_idx, end_idx))
        metadata = self._generate_experience_meta(batch_global_indexes, data_columns)
        return metadata

    def _scan_data_status(self, data_columns: List[str], global_step: int, task_name: str, get_n_samples: bool) -> List[int]:
        # 获取行和列掩码
        row_mask, col_mask = self.generate_data_status_mask(data_columns, global_step, task_name)
        logger.info(f"row_mask, col_mask: {row_mask, col_mask}")

        # 提取关注的数据状态子集
        logger.info(f"self.data_production_status: {self.data_production_status}")
        data_status_of_interest = self.data_production_status[:, col_mask]
        logger.info(f"data_status_of_interest: {data_status_of_interest}")

        # 使用torch.all向量化检查替代求和比较
        all_columns_ready = torch.all(data_status_of_interest, dim=1)

        # 结合行掩码筛选符合条件的样本
        ready_mask = all_columns_ready & row_mask

        if get_n_samples and self.num_n_samples > 1:
            # 重塑为组视图并检查组完整性
            group_all_ready = torch.all(ready_mask.view(-1, self.num_n_samples), dim=1)

            # 获取完整就绪的组索引
            ready_group_indices = group_all_ready.nonzero(as_tuple=False).flatten()

            # 计算所有样本索引
            sample_offset = torch.arange(self.num_n_samples, device=self.device)
            ready_for_consume_idx = (
                    ready_group_indices.unsqueeze(1) * self.num_n_samples + sample_offset
            ).flatten().tolist()

            return ready_for_consume_idx
        else:
            ready_for_consume_idx = torch.nonzero(ready_mask, as_tuple=False).flatten().tolist()
            logger.info(f"ready_for_consume_idx: {ready_for_consume_idx}")

            return ready_for_consume_idx

    def _register_default_policy(self) -> None:
        """
        register and offer several default sampling policies, including:
            node-irrelevant policies:
            1. random_strategy: get random index from global usable indexes
            2. dp_token_load_balancing_strategy: get index using karmarkar_karp strategy across DP
            3. similar_seqlen_strategy: get index around certain specified seqlen
            node-relevant policies:
            4. storage_unit_load_balancing_strategy: get index from different storage nodes using the round-robin strategy
        """
        self.schedule_policies = {"random_strategy": random_strategy,
                                  "dp_token_load_balancing_strategy": dp_token_load_balancing_strategy,
                                  "similar_seqlen_strategy": similar_seqlen_strategy,
                                  "storage_unit_load_balancing_strategy": storage_unit_load_balancing_strategy,
                                  }

    def _register_customized_policy(self, policy_name: str, policy_func: Callable) -> None:
        """
        register customized sampling policy as users may need
        """
        if not isinstance(policy_name, str):
            raise TypeError("ERROR: policy name must be a string.")

        if not callable(policy_func):
            raise TypeError("ERROR: policy func must be a callable function.")

        if policy_name in self.schedule_policies:
            raise ValueError(f"ERROR: {policy_name} has already been registered in default policies.")

        self.schedule_policies[policy_name] = policy_func

    def _run_schedule_policy(self, policy_name: str, ready_for_consume_idx: List[int], experience_count: int, *args,
                             **kwargs) -> Optional[List[int]]:
        """
        run the scheduler policy based on policy_name, ready_for_consume_idx, required experience_count
        now the schedule process is called by trainer and sample indexes for all DPs are decided altogether
        policy should not be bothered by get_n_samples requirement
        """
        if len(ready_for_consume_idx) < experience_count:
            # logger.info('Error: not enough data to consume yet.')
            return None

        assert len(ready_for_consume_idx) % experience_count == 0

        return self.schedule_policies[policy_name](ready_for_consume_idx=ready_for_consume_idx,
                                                   experience_count=experience_count, *args, **kwargs)

    def _generate_experience_meta(self, global_indexes: List[int], data_columns: List[str]) -> ExperienceMeta:
        # 根据给定的global index，查找self.global_index_local_index_mapping和self.global_index_storage_mapping，确定对应
        # 存储节点的地址，并构建ExperienceMeta
        global_arr = np.array(global_indexes)
        storage_ranks = self.global_index_storage_mapping[global_arr]
        local_indexes = self.global_index_local_index_mapping[global_arr]

        return ExperienceMeta(
            global_indexes=global_indexes,
            columns=data_columns,
            local_indexes=local_indexes.tolist(),
            storage_unit_ranks=storage_ranks.tolist()
        )

    def _update_production_status(self, indexes, column):
        # 更新数据生产状态矩阵
        for i, index in enumerate(indexes):
            if column[i] not in self.column_name_mapping:
                # 注册新列
                new_col_idx = len(self.column_name_mapping)
                self.column_name_mapping[column[i]] = new_col_idx

                # 检查生产状态矩阵是否需要扩展
                current_cols = self.data_production_status.shape[1]
                if new_col_idx > current_cols:
                    # 至少增加10列
                    add_cols = max(10, new_col_idx - current_cols + 1)
                    new_columns = torch.zeros(
                        (self.storage_size, add_cols),
                        dtype=torch.int8,
                        device=self.data_production_status.device
                    )
                    self.data_production_status = torch.cat(
                        [self.data_production_status, new_columns], dim=1
                    )

            index = torch.tensor(index, dtype=torch.long)
            col_idx = self.column_name_mapping[column[i]]

            self.data_production_status[index, col_idx] = 1

    def _init_zmq_socket(self):
        # 建立3个ZMQ服务端口，分别用于 ①注册发现 ② 接收Client的数据读写请求 ③ 接收Storage发送的状态更新信号
        self.zmq_context = zmq.Context()

        self._node_ip = get_node_ip()
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
            }
        )

    def _wait_connection(self):
        # 等待所有存储实例握手;client无需握手以支持动态扩缩容
        # 参考zmq_communication.py中的实现
        # TODO(zjj): 考虑是否需要重传（假设存在Storage没有收到ACK的情况）
        connected_storage_units = set()
        while len(connected_storage_units) < self.num_storage_units:
            identity, msg = self.handshake_socket.recv_multipart()
            msg = ZMQMessage.deserialize(msg)
            if msg.request_type == ZMQRequestType.HANDSHAKE:
                connected_storage_units.add(msg.sender_id)
                ack_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.HANDSHAKE_ACK,
                    sender_id=self.controller_id,
                    body={},
                ).serialize()
                self.handshake_socket.send_multipart([identity, ack_msg])
                logger.info(f"Controller send handshake ack successful!")
        self.storage_units = connected_storage_units

    def _start_process_request(self):
        # 拉起处理任务
        self.process_update_data_status_thread = Thread(target=self._update_data_status,
                                                        name="TransferQueueControllerProcessUpdateDataStatusThread",
                                                        daemon=True)
        self.process_update_data_status_thread.start()

        self.process_request_thread = Thread(target=self._process_request,
                                             name="TransferQueueControllerProcessRequestThread",
                                             daemon=True)
        self.process_request_thread.start()

    def _process_request(self):
        # 包含_get_meta、查询当前iteration是否消费完毕等
        while True:
            # ROUTER套接字接收多部分消息
            identity, request_bytes = self.request_handle_socket.recv_multipart()
            request = ZMQMessage.deserialize(request_bytes)

            if request.request_type == ZMQRequestType.GET_PROMPT_META:
                params = request.body

                metadata = self._get_prompt_metadata(
                    data_columns=params['data_columns'],
                    experience_count=params['experience_count'],
                    global_step=params['global_step'],
                    n_samples_per_prompt=params['n_samples_per_prompt'],
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_PROMPT_META_RESPONSE,
                    sender_id=self.controller_id,
                    body={
                        'status': 'SUCCESS',
                        'metadata': metadata
                    }
                )
            elif request.request_type == ZMQRequestType.GET_META:
                params = request.body
                # 处理元数据请求
                logger.info(f"Controller prepare get metadata...")
                metadata = self._get_metadata(
                    data_columns=params['data_columns'],
                    experience_count=params['experience_count'],
                    global_step=params['global_step'],
                    dp_world_size=params['dp_world_size'],
                    task_name=params['task_name'],
                    get_n_samples=params.get('get_n_samples', False),
                    schedule_policy=params.get('schedule_policy', 'DP_balance')
                )

                # 构建响应消息
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request.sender_id,
                    body={
                        'status': 'SUCCESS',
                        'metadata': metadata
                    }
                )
            elif request.request_type == ZMQRequestType.CHECK_CONSUMPTION:
                # 消费状态检查
                params = request.body
                global_step = params['global_step']

                consumer_status = self._get_consumer_status(params['task_name'])
                start_idx = (global_step % self.num_global_batch) * self.global_batch_size * self.num_n_samples
                end_idx = start_idx + self.global_batch_size * self.num_n_samples
                batch_status = consumer_status[start_idx:end_idx]
                consumed = torch.all(batch_status == 1).item()

                # 构建响应消息
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request.sender_id,
                    body={
                        'status': 'SUCCESS',
                        'global_step': global_step,
                        'consumed': consumed,
                    }
                )
            self.request_handle_socket.send_multipart([
                identity,
                response_msg.serialize()
            ])
            logger.warning(f"Controller request_handle_socket send_multipart successful!")

    def _update_data_status(self):
        # 用于接受来自storage的数据状态更新信息
        while True:
            logger.warning(f"Prepare _update_data_status...")
            identity, request_bytes = self.data_status_update_socket.recv_multipart()
            logger.info(f"Controller recv update_data_status requeset!")
            request = ZMQMessage.deserialize(request_bytes)

            if request.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE:
                params = request.body
                message_data = params.get("message", {})
                row_idx = message_data.get("row_idx")
                col_idx = message_data.get("col_idx")

                # 更新数据生产状态
                print(f"row_idx, col_idx: {row_idx, col_idx}")
                self._update_production_status(row_idx, col_idx)
                logger.info("Controller update production status successful!")

                # 发送确认响应
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Data update acknowledged from controller {self.controller_id}"
                    }
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])
                logger.info(f"Controller send DATA_UPDATE_ACK successful!")
            elif request.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR:
                # 处理数据更新错误
                error_msg = request.body.get("message", "Unknown error")
                print(f"Data update error from storage: {error_msg}")

                # 发送错误确认响应
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Error notification acknowledged from controller {self.controller_id}"
                    }
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])

    def get_zmq_server_info(self) -> ZMQServerInfo:
        return self.zmq_server_info

    def clear(self, global_steps: Union[int, List[int]]):
        # 清空对应global batch的数据
        pass


class TransferQueueStorage(ABC):
    # TODO
    # 提供一个Storage基础能力的抽象，可被各种分布式存储后端实现，作为存储面的统一管理
    # 更一般地，可以提供一个general的kv存储，TransferQueueController实现global index和分布式存储的key的查找表，直接调用底层分布式
    # 存储的接口进行数据读写
    def __init__(self):
        pass

    def put(self):
        pass

    def get(self):
        pass


@ray.remote(max_concurrency=50, num_cpus=4)
class TransferQueueStorageSimpleUnit(TransferQueueStorage):
    def __init__(self, storage_unit_id: str, storage_size: int):
        super().__init__()
        # TODO: @zhongjianjun 讨论 Storage 和 Controller 的各个 socket 是否需要使用枚举
        # self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4()}"
        self.storage_unit_id = storage_unit_id  # FIXME 临时回滚：后续不在主控指定ID @zhognjianjun
        self.storage_size = storage_size
        self.controller_info = None
        self.current_size = 0

        # TODO: 数据结构需明确，提供一个类 @congzhen, 并加速读写操作，当前put get效率太低
        self.experience_data = {}  # {row_uuid: {column_id: value}}

        self.zmq_server_info = ZMQServerInfo.create(
            role=TransferQueueRole.STORAGE,
            id=str(self.storage_unit_id),
            ip=get_node_ip(),
            ports={"put_get_socket": get_free_port()}
        )
        self._init_zmq_socket()

    def _init_zmq_socket(self):
        """
        ############################################################################################################################################################
        socket类型                           zmq类型        访问类型               request_type                    含义
        ############################################################################################################################################################
        self.controller_handshake_socket    DEALER         send                 HANDSHAKE                       向所有的controller发送handshake请求
        self.controller_handshake_socket    DEALER         recv                 HANDSHAKE_ACK                   接收到controller响应handshake成功的信号
        ############################################################################################################################################################
        self.data_status_update_socket      DEALER         send                 NOTIFY_DATA_UPDATE              向所有的controller广播通知数据更新的信息
        self.data_status_update_socket      DEALER         recv                 NOTIFY_DATA_UPDATE_ACK          接收到所有controller反馈的数据更新确认接收信息
        self.data_status_update_socket      DEALER         send                 NOTIFY_DATA_UPDATE_ERROR        向所有的controller广播通知数据更新错误的信息
        ############################################################################################################################################################
        self.put_get_socket                 ROUTER         recv_multipart       GET                             接收到client发送的数据读取请求
        self.put_get_socket                 ROUTER         recv_multipart       PUT                             接收到client发送的数据写入请求
        self.put_get_socket                 ROUTER         send_multipart       PUT_GET_OPERATION_ERROR         接收到client发送的数据读写请求，处理过程中发现非PUT/GET请求
        self.put_get_socket                 ROUTER         send_multipart       PUT_GET_ERROR                   接收到client发送的数据读写请求，处理过程出现错误
        self.put_get_socket                 ROUTER         send_multipart       PUT_ACK                         接收到client发送的数据写入请求，写入成功并返回数据写入相关信息
        self.put_get_socket                 ROUTER         send_multipart       PUT_FULL_ERROR                  接收到client发送的数据写入请求，数据条目数超过最大限制报错
        self.put_get_socket                 ROUTER         send_multipart       PUT_ERROR                       接收到client发送的数据写入请求，处理过程出现错误
        self.put_get_socket                 ROUTER         send_multipart       GET_ACK                         接收到client发送的数据读取请求，读取成功并返回读取数据结果
        self.put_get_socket                 ROUTER         send_multipart       GET_ERROR                       接收到client发送的数据读取请求，处理过程出现错误
        ############################################################################################################################################################
        """
        self.zmq_context = zmq.Context()

        # controller_handshake_socket用于和controller握手建链, 作为客户端(DEALER)发送请求, 并接收controller确认握手成功的ACK信息
        self.controller_handshake_sockets: Dict[str, zmq.Socket] = {}

        # data_status_update_socket用于和controller广播数据更新信息, 作为客户端(DEALER)广播信息, 并接收controller确认拿到更新信息的ACK信息
        self.data_status_update_socket: Dict[str, zmq.Socket] = {}

        # put_get_socket用于接收client发起的数据读/写请求，作为服务端(ROUTER)处理请求, 并反馈处理结果
        self.put_get_socket_address = self.zmq_server_info.to_addr("put_get_socket")
        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER)
        self.put_get_socket.bind(self.put_get_socket_address)

        """
        获取self.put_get_socket绑定的地址
        endpoint = self.put_get_socket.getsockopt(zmq.LAST_ENDPOINT)
        if endpoint:
            str = endpoint.decode("utf-8")
            match = re.match(r"tcp://(.+):(\d+)", endpoint_str)
            if match:
                ip = match.group(1)
                port = match.group(2)
        """

    def register_controller_info(self, controller_infos: Dict[str, ZMQServerInfo]):
        self.controller_infos = controller_infos
        self._init_zmq_socket_using_controller_infos()
        self._connect_to_controller()
        self._start_process_put_get()

    def _init_zmq_socket_using_controller_infos(self):
        for controller_id in self.controller_infos.keys():
            self.controller_handshake_sockets[controller_id] = create_zmq_socket(self.zmq_context, zmq.DEALER, identity=f"{self.storage_unit_id}-controller_handshake_sockets-{uuid4()}".encode())
            self.data_status_update_socket[controller_id] = create_zmq_socket(self.zmq_context, zmq.DEALER, identity=f"{self.storage_unit_id}-data_status_update_socket-{uuid4()}".encode())

    def _connect_to_controller(self):
        # TODO(zjj): 单个DEALER连接多个ROUTER，默认做负载均衡，因此使用多个socket握手。考虑使用单个socket指定连接的ROUTER
        """Connect to all controllers"""
        connected_controllers = set()

        # Create zmq poller for controller-storage handshake confirmation
        poller = zmq.Poller()

        for controller_id, controller_info in self.controller_infos.items():
            self.controller_handshake_sockets[controller_id].connect(controller_info.to_addr("handshake_socket"))
            logger.info(f"Controller id #{controller_id} connection successful")

            # handshake with controllers
            self.controller_handshake_sockets[controller_id].send(
                ZMQMessage.create(
                    request_type=ZMQRequestType.HANDSHAKE,
                    sender_id=self.zmq_server_info.id,
                    body={
                        "storage_unit_id": self.storage_unit_id,
                        "storage_size": self.storage_size,
                    }
                ).serialize()
            )
            poller.register(self.controller_handshake_sockets[controller_id], zmq.POLLIN)

        start_time = time.time()
        while len(connected_controllers) < len(
                self.controller_infos) and time.time() - start_time < CONTROLLER_STORAGE_HANDSHAKE_TIMEOUT:
            socks = dict(poller.poll(POLLER_MAX_SIZE))
            logger.info(f"Prepare for handshake...")
            for controller_handshake_socket in self.controller_handshake_sockets.values():
                if controller_handshake_socket in socks:
                    msg = ZMQMessage.deserialize(controller_handshake_socket.recv())
                    if msg.request_type == ZMQRequestType.HANDSHAKE_ACK:
                        connected_controllers.add(msg.sender_id)
                        # TODO: 各种id都是字符串
                        logger.info(f"Controller id # {self.controller_handshake_sockets[int(msg.sender_id)]} and storage "
                                    f"id # {self.zmq_server_info.id} handshake successful.")

        if len(connected_controllers) < len(self.controller_infos):
            print(f"Warning: After controller handshake from TransferQueueStorageSimpleUnit_{self.zmq_server_info.id}, "
                  f"only connected to {len(connected_controllers)} out of {len(self.controller_infos)} controllers.")

    def _start_process_put_get(self):
        self.process_put_get_thread = Thread(
            target=self._process_put_get,
            name=f"TransferQueueStorageSimpleUnitProcessPutGetThread_{self.zmq_server_info.id}",
            daemon=True
        )
        self.process_put_get_thread.start()

    def _process_put_get(self):
        poller = zmq.Poller()
        poller.register(self.put_get_socket, zmq.POLLIN)

        # 二进制信号量, 形成互斥锁
        self._semaphore = multiprocessing.Semaphore(1)

        while True:
            socks = dict(poller.poll(POLLER_MAX_SIZE))
            logger.info(f"socks: {socks}")
            if self.put_get_socket in socks:
                identity, msg_bytes = self.put_get_socket.recv_multipart()
                try:
                    msg = ZMQMessage.deserialize(msg_bytes)
                    operation = msg.request_type
                    logger.error(f"Storage get process_put_get operation: {operation}")

                    if operation == ZMQRequestType.PUT_DATA:
                        response_msg = self._handle_put(msg)
                    elif operation == ZMQRequestType.GET_DATA:
                        response_msg = self._handle_get(msg)
                    else:
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.PUT_GET_OPERATION_ERROR,
                            sender_id=self.zmq_server_info.id,
                            body={
                                "message": f"TransferQueueStorageSimpleUnit_{self.zmq_server_info.id} occur "
                                            f"invalid operation: {operation}"
                            }
                        )
                except Exception as e:
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.PUT_GET_ERROR,
                        sender_id=self.zmq_server_info.id,
                        body={
                            "message": f"TransferQueueStorageSimpleUnit_{self.zmq_server_info.id} occur error in "
                                    f"processing put/get, detail error message: {str(e)}"
                        }
                    )
                self.put_get_socket.send_multipart([identity, response_msg.serialize()])

    def _handle_put(self, data_parts: ZMQMessage):
        ####### data_parts structure ######
        ##   row   #      row name       ##
        ##   col   #    column name      ##
        ##  item   #   real data item    ##
        ###################################
        self._semaphore.acquire()
        try:
            # TODO：这里的row col item等较难理解，和ExperienceMeta中给出的信息有gap，需优化 @congzhen
            print(f"storage get put data_parts: {data_parts}")
            data_parts = data_parts.body["data_parts"]
            row_idx = data_parts.get("row", None)
            col_idx = data_parts.get("col", None)
            item = data_parts.get("item", None)

            new_rows = [row for row in row_idx if row not in self.experience_data]
            if self.current_size + len(new_rows) > self.storage_size:
                logger.warning(f"WARNING: Storage unit is full! Current: {self.current_size}, Max: {self.storage_size}, Requested: {len(new_rows)}")

                return ZMQMessage.create(
                    request_type=ZMQRequestType.PUT_FULL_ERROR,
                    sender_id=self.zmq_server_info.id,
                    body={
                        "message": f"TransferQueueStorageSimpleUnit_{self.zmq_server_info.id} is full, "
                                    f"current data can not be put into this storage unit, "
                                    f"storage max size: {self.storage_size}."
                    }
                )

            # TODO: 效率优化 @congzhen
            for idx, row in enumerate(row_idx):
                if row not in self.experience_data:
                    self.experience_data[row] = {}
                    self.current_size += 1

                self.experience_data[row][col_idx[idx]] = item[idx]
            print(f"self.experience_data: {self.experience_data}")

            # after put operation finish, send a message to the client
            logger.warning(f"Storage prepare send PUT_DATA_RESPONSE to client...")
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": {
                        "storage_unit_size": self.storage_size,
                        "current_size": self.current_size,
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        # TODO：这里的row col item等较难理解，和ExperienceMeta中给出的信息有gap，需优化 @congzhen
                        "zmq_server_info": self.zmq_server_info,
                    }
                }
            )
            logger.info(f"Storage get PUT_DATA_RESPONSE successful!")
            # broadcast data update message to controllers
            self._notify_data_update(row_idx, col_idx)
            return response_msg
        except Exception as e:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to put data into TransferQueueStorageSimpleUnit_{self.zmq_server_info.id}, "
                                f"detail error message: {str(e)}"
                }
            )
        finally:
            self._semaphore.release()

    def _notify_data_update(self, row_idx, col_idx):
        # Create zmq poller for notify data update information
        # TODO： 加上global idx @congzhen
        # 这个notify过程的执行效率需要好好分析一下
        poller = zmq.Poller()

        # Connect data status update socket to all controllers
        for controller_id, controller_info in self.controller_infos.items():
            # TODO @zhangyebin controller ZMQServerInfo.ports需要包含"controller_storage_notify_data_update_socket"信息
            data_status_update_socket = self.data_status_update_socket[controller_id]
            logger.info(f"Storage unit Prepare connect data_status_update_socket...| controller: {controller_info}")
            data_status_update_socket.connect(controller_info.to_addr("data_status_update_socket"))
            logger.info(f"Storage unit data_status_update_socket connect successful!")
            try:
                poller.register(data_status_update_socket, zmq.POLLIN)
                data_status_update_socket.send(
                    ZMQMessage.create(
                        request_type=ZMQRequestType.NOTIFY_DATA_UPDATE,
                        sender_id=self.zmq_server_info.id,
                        body={
                            "message": {
                                "storage_unit_size": self.storage_size,
                                "current_size": self.current_size,
                                "row_idx": row_idx,
                                "col_idx": col_idx,
                                # TODO：这里的row col item等较难理解，和ExperienceMeta中给出的信息有gap，需优化 @congzhen
                                "zmq_server_info": self.zmq_server_info,
                            }
                        }
                    ).serialize()
                )
            except Exception as e:
                logger.warning(f"Storage unit data_status_update_socket error")
                data_status_update_socket.send(
                    ZMQMessage.create(
                        request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR,
                        sender_id=self.zmq_server_info.id,
                        body={
                            "message": f"Failed to notify data status update information from "
                                    f"TransferQueueStorageSimpleUnit_{self.zmq_server_info.id}, detail error message: {str(e)}"
                        }
                    ).serialize()
                )

        response_controllers = set()
        start_time = time.time()
        while len(response_controllers) < len(
                self.controller_infos) and time.time() - start_time < CONTROLLER_DATA_UPDATE_RESPONSE_TIMEOUT:
            socks = dict(poller.poll(POLLER_MAX_SIZE))
            for data_status_update_socket in self.data_status_update_socket.values():
                if data_status_update_socket in socks:
                    msg = ZMQMessage.deserialize(data_status_update_socket.recv())
                    if msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                        # TODO @zhangyebin controller在向storage发送NOTIFY_DATA_UPDATE_ACK时, 需要同步反馈ZMQServerInfo.serder_id
                        response_controllers.add(msg.sender_id)

        if len(response_controllers) < len(self.controller_infos):
            print(f"Warning: After data status update in TransferQueueStorageSimpleUnit_{self.zmq_server_info.id}, "
                  f"only get {len(response_controllers)} out of {len(self.controller_infos)} ACK responses from controllers.")

    def _handle_get(self, data_parts: ZMQMessage):
        ####### data_parts structure #######
        ##   row  #       row name        ##
        ##   col  #     column name       ##
        ####################################
        try:
            logger.info(f"_handle_get data_parts: {data_parts}")
            row = data_parts.body.get("local_indexes", None)
            col = data_parts.body.get("columns", None)

            # TODO: 效率优化 @congzhen
            result_data = {}
            for row_idx in row:
                if row_idx in self.experience_data:
                    row_data = {}
                    for col_idx in col:
                        if col_idx in self.experience_data[row_idx]:
                            row_data[col_idx] = self.experience_data[row_idx][col_idx]
                        else:
                            row_data[col_idx] = None
                            logger.warning(f"Column {col_idx} not found in row {row_idx}")
                    result_data[row_idx] = row_data
                else:
                    row_data = {col_idx: None for col_idx in col}
                    result_data[row_idx] = row_data
                    logger.warning(f"Row {row_idx} not found in storage")

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": {
                        "status": "SUCCESS",
                        "data": result_data,
                        "zmq_server_info": self.zmq_server_info,
                    }
                }
            )

        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to get data from TransferQueueStorageSimpleUnit_{self.zmq_server_info.id}, "
                               f"detail error message: {str(e)}"
                }
            )
        return response_msg

    def get_zmq_server_info(self) -> ZMQServerInfo:
        return self.zmq_server_info


class TransferQueueClient:
    def __init__(
            self,
            client_id: str,
            controller_infos: Union[ZMQServerInfo, Dict[Any, ZMQServerInfo]],
            storage_infos: Union[ZMQServerInfo, Dict[Any, ZMQServerInfo]],
            dp_world_size: Optional[int] = None,
            num_dp_groups: Optional[int] = None,
            dp_rank: Optional[int] = None,
            rank_id: Optional[int] = None,
    ):

        self.client_id = client_id  # 方便debug，记录请求来源
        self.controller_infos = controller_infos
        self.storage_infos = storage_infos
        self.dp_world_size = dp_world_size
        self.num_dp_groups = num_dp_groups
        self.dp_rank = dp_rank
        self.rank_id = rank_id

        self._init_zmq_socket()

    def _init_zmq_socket(self):
        # 动态连接
        self.zmq_context = zmq.Context()
        self.controller_id_to_socket = {}  # {id: socket}
        self.storage_id_to_socket = {}  # {id: socket}

        def _connect_to_server(server_infos: Union[ZMQServerInfo, Dict[Any, ZMQServerInfo]], role: TransferQueueRole,
                               socket_name=""):
            server_infos = server_infos if isinstance(server_infos, dict) else {0: server_infos}
            for info in server_infos.values():
                try:
                    address = f"tcp://{info.ip}:{info.ports.get(socket_name)}"
                    sock = create_zmq_socket(
                        self.zmq_context, zmq.DEALER, identity=f"{self.client_id}_to_{info.id}_{uuid4()}".encode()
                    )
                    sock.connect(address)
                    sock_mapping = self.controller_id_to_socket if role == TransferQueueRole.CONTROLLER else self.storage_id_to_socket
                    sock_mapping[info.id] = sock
                    logger.info(f"Client {self.client_id} connected to {role} {info.id} at {address}")
                except Exception as e:
                    logger.error(f"Failed to connected to {role} {info.id}: {e}")

        _connect_to_server(self.storage_infos, TransferQueueRole.STORAGE, socket_name="put_get_socket")
        _connect_to_server(self.controller_infos, TransferQueueRole.CONTROLLER, socket_name="request_handle_socket")

    def put(self, data: TensorDict, metadata: ExperienceMeta):
        """
        根据metadata将数据写入到对应的Storage后端

        参数:
        data: 要写入的数据，TensorDict格式
        metadata: 数据的元信息，包含索引和存储单元信息
        """
        # TODO：async包装 @huazhong
        # TODO： row col item不太好理解，和get_data的column、local indexes不对应 @huazhong
        if not metadata or metadata.size == 0:
            raise ValueError("metadata cannot be None or empty")
        # TODO: data不具有global indexes，直接从metadata拿取 @huazhong
        data["global_indexes"] = torch.tensor(metadata.global_indexes)
        # 1. 按存储单元分组数据
        storage_data = {}  # 存储单元ID -> 对应的数据部分

        # 利用ExperienceMeta的storage_unit_index_map进行数据分组
        for su_rank, index_data in metadata.storage_unit_index_map.items():
            storage_data[su_rank] = {
                'row': [],
                'col': [],
                'item': [],
            }

            # 为每个存储单元准备要写入的数据
            for i, global_idx in enumerate(index_data['global_indexes']):
                local_idx = index_data['local_indexes'][i]

                # 将数据按列拆分
                for col in index_data['columns']:
                    # 获取该列的数据
                    idx = (data['global_indexes'] == global_idx).nonzero().item()
                    item = data[col][idx]

                    # 添加到存储单元的数据列表中
                    storage_data[su_rank]['row'].append(local_idx)
                    storage_data[su_rank]['col'].append(col)
                    storage_data[su_rank]['item'].append(item)

        # 2. 向每个存储单元发送数据
        for su_rank, su_data in storage_data.items():
            print(f"client create su_data: {su_data}")
            # 获取对应的存储单元socket
            if str(su_rank) not in self.storage_id_to_socket:
                raise RuntimeError(f"Storage unit {su_rank} not available")
            storage_socket = self.storage_id_to_socket[str(su_rank)]

            # 创建ZMQ消息
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA,
                sender_id=self.client_id,
                receiver_id=str(su_rank),
                body={
                    'data_parts': su_data
                }
            ).serialize()

            try:
                # 发送请求
                # TODO(hz): 性能优化，多su_rank的数据并发发送

                storage_socket.send(request_msg)

                # 接收响应
                serialized_body = storage_socket.recv()
                response_msg = ZMQMessage.deserialize(serialized_body)

                if response_msg.request_type != ZMQRequestType.PUT_DATA_RESPONSE or not response_msg.body[
                    'message']:
                    raise RuntimeError(
                        f"Failed to put data to storage unit {su_rank}: {response_msg.body.get('message', 'Unknown error')}")
            except Exception as e:
                raise RuntimeError(f"Error in put to storage unit {su_rank}: {str(e)}")

    # DEPRECATED：第一阶段无需设计此函数
    # def get(self, data_columns:List[str], experience_count:int, dp_world_size:int, num_dp_groups:int=None,  rank_id:int=None,  get_n_samples=False, schedule_policy:str='DP_balance', *args, **kwargs) -> (TensorDict, ExperienceMeta, int):
    #     # 获取对应的meta data和数据，封装了get_meta和get_data两个步骤
    #     # 这里DP相关的配置用来支持不同的数据获取方式：
    #     # 方式1. 每个worker进程自己向主控发起数据获取请求，dp_rank用来区分来自不同dp域的请求，dp_size作为计数器，统计一个batch的数据是否被DP域内所有rank拿走过一遍
    #
    #     metadata, current_global_step = self.get_meta(data_columns, experience_count, dp_world_size, num_dp_groups, rank_id)
    #     data = self.get_data(metadata)
    #     return data, metadata, current_global_step

    def get_meta(
            self,
            data_columns: List[str],
            experience_count: int,
            global_step:int,
            dp_world_size: int,
            get_n_samples: bool = False,
            task_name: str = None,
            schedule_policy: str = 'DP_balance',
            *args,
            **kwargs,
    ) -> ExperienceMeta:
        """
        只有主控拿到全部数据的meta data，再通过Dispatch分发metadata给各个worker，此时不需要指定dp_rank和dp_size
        这时直接在主控中调用get_meta即可，无需调用get
        内部逻辑：
        1. 建立与Controller的连接
        2. 向Controller发送数据获取请求，获取meta data
        NOTE(hz): 按照讨论，controller跟wg绑定，get_meta在主控调用，主控可以拿到全局的wg和controller，意味着client只跟唯一的controller绑定
        """
        # metadata = ExperienceMeta()
        # return metadata, current_global_step
        # 选择一个控制器进行通信（这里简化为使用第一个可用的控制器）
        # TODO：async包装 @huazhong
        if not self.controller_id_to_socket:
            raise RuntimeError("No controller available")

        controller_id, controller_socket = next(iter(self.controller_id_to_socket.items()))

        # 创建ZMQ消息，设置receiver_id为控制器ID
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_META,
            sender_id=self.client_id,
            receiver_id=controller_id,
            body={
                'data_columns': data_columns,
                'experience_count': experience_count,
                'global_step': global_step,
                'dp_world_size': dp_world_size,
                'get_n_samples': get_n_samples,
                'task_name': task_name,
                'schedule_policy': schedule_policy
            }
        ).serialize()

        try:
            # 发送消息
            logger.info(f"Clinet prepare send get meta request to Controller #{controller_id}")
            controller_socket.send(request_msg)

            # 接收响应
            serialized_body = controller_socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_body)
            logger.info(f"Clinet get datameta response: {response_msg}")

            if response_msg.request_type == ZMQRequestType.GET_META_RESPONSE.value and response_msg.body[
                'status'] == 'SUCCESS':
                metadata = response_msg.body['metadata']
                return metadata
            else:
                raise RuntimeError(f"Failed to get metadata: {response_msg.body.get('message', 'Unknown error')}")
        except Exception as e:
            raise RuntimeError(f"Error in get_meta: {str(e)}")

    def get_data(self, metadata: ExperienceMeta) -> TensorDict:
        """
        1. 根据metadata，向Storage Unit发送数据读取请求，获得数据
        2. 并根据metadata的global_indexes的顺序把从不同Storage Unit获取的数据合并到一个TensorDict里
        """
        # TODO：async包装 @huazhong
        if not metadata or metadata.size == 0:
            return TensorDict({}, batch_size=0)

        storage_data = {}  # global_index: {col1: value, col2: value, ...}
        # 向每个存储单元发送请求
        # TODO(hz): parallelize this
        logger.info(f"client get data: {metadata}")
        for su_rank, index_data in metadata.storage_unit_index_map.items():
            # TODO(hz): su_rank -> su_id
            if str(su_rank) not in self.storage_id_to_socket:
                raise RuntimeError(f"Storage unit {su_rank} not available")
            socket = self.storage_id_to_socket[str(su_rank)]

            global_indexes = index_data['global_indexes']
            local_indexes = index_data['local_indexes']
            columns = index_data['columns']

            # 创建ZMQ消息
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA,
                sender_id=self.client_id,
                receiver_id=str(su_rank),
                body={
                    'local_indexes': local_indexes,
                    'columns': columns
                }
            ).serialize()

            try:
                # 发送请求
                logger.info(f"Prepare send get data request to storage")
                socket.send(request_msg)

                # 接收响应
                serialized_body = socket.recv()
                response_msg = ZMQMessage.deserialize(serialized_body)
                logger.info(f"get data response_msg: {response_msg}")

                if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE and response_msg.body[
                    'message']['status'] == 'SUCCESS':
                    # 存储该存储单元返回的数据
                    su_data = response_msg.body['message']['data']
                    # 将数据与原始索引关联
                    for idx, local_idx in enumerate(local_indexes):
                        global_idx = global_indexes[idx]
                        if global_idx not in storage_data:
                            storage_data[global_idx] = {}
                        for col in columns:
                            storage_data[global_idx][col] = su_data[local_idx][col]
                else:
                    raise RuntimeError(
                        f"Failed to get data from storage unit {su_rank}: {response_msg.body.get('message', 'Unknown error')}")
            except Exception as e:
                raise RuntimeError(f"Error getting data from storage unit {su_rank}: {str(e)}")

        # # 2. 按global_indexes顺序合并数据
        # ordered_data = {col: [] for col in metadata.columns}
        # for global_idx in metadata.global_indexes:
        #     if global_idx not in storage_data:
        #         raise RuntimeError(f"Data for global index {global_idx} not found")
        #     for col in metadata.columns:
        #         ordered_data[col].append(storage_data[global_idx][col])

        # 3. storage_data -> tensordict
        # 按列组织数据
        indexes = []
        ordered_data = {col: [] for col in metadata.columns}

        # 按照storage_data写入的顺序遍历
        for idx in storage_data.keys():
            indexes.append(idx)
            for col in metadata.columns:
                ordered_data[col].append(storage_data[idx][col])

        tensor_data = {
            col: torch.tensor(v)
            for col, v in ordered_data.items()
        }

        # 为tensor_data每个tensor的batch维增加索引信息
        tensor_data['global_indexes'] = torch.tensor(indexes)

        # 创建TensorDict
        """
        Example:
        tensor_dict = {
            "prompt_token_ids": torch.randn(4,128),
            "response_token_ids": torch.randn(4, 256),
            "global_indexes": torch.Tensor([7, 4, 6, 5]), # prompt_token_ids/response_token_ids每条样本对应的global_idx
        }
        NOTE: 为什么要记录global_indexes？
              - 由于在具体任务中可能会对batch进行rearange操作，会调整batch内样本顺序，global_indexes可以记录这种顺序的调整，便于后续把数据放回
                Storage Unit时从过metadata（根据global_indexes）找到对应的Storage Unit
        """
        tensor_dict = TensorDict(tensor_data, batch_size=len(storage_data))
        return tensor_dict

    def put_prompts(self, data: TensorDict, data_columns: list[str], global_step: int = 0,
                    n_samples_per_prompt: int = 1):
        """
        向controller发送消息构造metadata，再根据metadata写入storage units

        参数:
        data: 包含prompt数据的TensorDict
        global_step: 当前全局步数
        n_samples_per_prompt: 每个prompt要生成的样本数

        可以任意选一个controller，因为写入后storage会将数据状态更新信息广播给所有controller
        """
        # TODO: async
        if not self.controller_id_to_socket:
            raise RuntimeError("No controller available")

        controller_id, controller_socket = next(iter(self.controller_id_to_socket.items()))

        # 1. 构造请求消息向controller获取metadata
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_PROMPT_META,
            sender_id=self.client_id,
            receiver_id=controller_id,
            body={
                "data_columns": data_columns,
                "experience_count": data.batch_size[0],
                "global_step": global_step,
                "n_samples_per_prompt": n_samples_per_prompt,
            }
        ).serialize()

        # 2. 发送请求到任意一个controller
        controller_socket.send(request_msg)

        # 3. 接收controller的响应
        response = controller_socket.recv()
        response_msg = ZMQMessage.deserialize(response)

        # 4. 解析响应中的metadata
        metadata = response_msg.body["metadata"]

        # 5. 根据metadata将数据写入storage units
        self.put(data, metadata)
        logger.info("Clinet put prompts finished!")

    # DEPRECATED：第一阶段无需设计此函数，后续dataloader等抽象作为recipe提供一种极致性能实现
    # def get_data_loader(self, data_columns:List[str], experience_count:int, dp_world_size:int, num_dp_groups:int=None,  rank_id:int=None) -> StreamDataLoader:
    #     # 构造迭代器将get过程进行抽象
    #     pass


    def check_current_step_consumption(self, global_step):
        # 检查当前global batch是否消耗完
        pass


# DEPRECATED：第一阶段无需设计此函数，后续dataloader等抽象作为recipe提供一种极致性能实现
# class StreamDataLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset: StreamingDataset):
#         self.dataset = dataset
#         super().__init__(dataset=self.dataset, collate_fn=_custom_collate)

# DEPRECATED：第一阶段无需设计此函数，后续dataloader等抽象作为recipe提供一种极致性能实现
# class StreamingDataset(IterableDataset):
#     def __init__(self, client_handler):
#         super().__init__()
#         self.client_handler = client_handler
#         pass
#
#     def __iter__(self):
#         while self.client_handler.check_current_step_consumption():
#             pass


def process_zmq_server_info(handlers: Dict[Any, Union[TransferQueueController, TransferQueueStorageSimpleUnit]]):
    server_info = {}
    for name, handlers in handlers.items():
        server_info[name] = ray.get(handlers.get_zmq_server_info.remote())

    return server_info
