# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import copy
import heapq
from typing import List, Tuple, Optional
import numpy
import torch
import torch.distributed as dist
from torch import Tensor

## ==  This file provides default scheduling policies that
# can be directly used by TransferQueue data system. == ##

def random_strategy(
        ready_for_consume_idx: List[int], experience_count: int, get_n_samples: bool, n_samples_per_prompt: int,
) -> Optional[List[int]]:
    """
    random sampling experience_count samples from global indexes ready_for_consume_idx
    input example:
        if get_n_samples: (group_num=3, group_size=4)
            ready_for_consume_idx could look like: [0, 1, 2, 3,   8, 9, 10, 11,   16, 17, 18, 19]
        else:
            ready_for_consume_idx could look like: [2, 5, 6]
    """
    if get_n_samples:
        assert len(ready_for_consume_idx) % n_samples_per_prompt == 0
        assert experience_count % n_samples_per_prompt == 0
        experience_count_n_samples = experience_count // n_samples_per_prompt

        group_ready_for_consume_idx = torch.tensor(ready_for_consume_idx, dtype=torch.int).view(-1,
                                                                                                n_samples_per_prompt)

        weights = torch.ones(group_ready_for_consume_idx.size(0))
        sampled_indexes_idx = torch.multinomial(weights, experience_count_n_samples, replacement=False).tolist()
        sampled_indexes = group_ready_for_consume_idx[sampled_indexes_idx].flatten().tolist()
    else:
        weights = torch.ones(len(ready_for_consume_idx))
        sampled_indexes_idx = torch.multinomial(weights, experience_count, replacement=False).tolist()
        sampled_indexes = [int(ready_for_consume_idx[i]) for i in sampled_indexes_idx]
    return sampled_indexes


def target_seq_len_strategy(
        ready_for_consume_idx: List[int], experience_count: int, get_n_samples: bool,
        n_samples_per_prompt: int, seq_len: Tensor, target_seq_len: int, method: Optional[str]='combined',
) -> Optional[List[int]]:
    """
    get index whose seq_len is around certain specified target_seq_len
    need to pass seq_len, which is the corresponding Tensor of seq len for ready_for_consume_idx
        Caution: when get_n_samples is True, we need to sample a group of experiences.
                however, the variance of seq len within each group could be large.
                it may not be appropriate to use closeness between one sample's seq_len in a group and target_len
                to measure the weights and process sampling.
        Currently, we allow three strategies "mean" "median" "combined"(consider variance) to calculate the weights
    """
    assert len(ready_for_consume_idx) == len(seq_len)
    if target_seq_len is None:
        raise ValueError("ERROR: target_seq_len cannot be None when using similar_seqlen_strategy.")
    elif not isinstance(seq_len, Tensor):
        raise ValueError("ERROR: seq_len must be a tensor of int.")
    else:
        pass

    if get_n_samples:
        assert len(ready_for_consume_idx) % n_samples_per_prompt == 0
        assert experience_count % n_samples_per_prompt == 0
        experience_count_n_samples = experience_count // n_samples_per_prompt
        group_ready_for_consume_idx = torch.tensor(ready_for_consume_idx,
                                                   dtype=torch.int).view(-1, n_samples_per_prompt)
        group_seq_len = seq_len.view(-1, n_samples_per_prompt).float()
        if method=='mean':
            group_seq_len_deviation = torch.abs(group_seq_len.mean(dim=1) - target_seq_len)
            # print(group_seq_len_deviation)
        elif method=='median':
            group_seq_len_deviation = torch.abs(group_seq_len.quantile(q=0.5, dim=1) - target_seq_len)
            # print(group_seq_len_deviation)
        else: # method=='combined':
            group_seq_len_mean_deviation = torch.abs(group_seq_len.mean(dim=1) - target_seq_len)
            # the smaller, the closer
            norm_group_seq_len_mean_deviation = group_seq_len_mean_deviation/(torch.max(group_seq_len_mean_deviation)+1e-5)
            group_seq_len_std_deviation = group_seq_len.std(dim=1)
            # the smaller, the seq len in the group is more even
            norm_group_seq_len_std_deviation = group_seq_len_std_deviation/(torch.max(group_seq_len_std_deviation)+1e-5)
            # simply used weighted values to generate sampling weights
            group_seq_len_deviation = 0.7 * norm_group_seq_len_mean_deviation + 0.3 * norm_group_seq_len_std_deviation
            # print(norm_group_seq_len_mean_deviation, * norm_group_seq_len_std_deviation, group_seq_len_deviation)

        # weights = torch.sigmoid(1 / (group_seq_len_deviation + 1e-3))
        weights = 1 - torch.exp(- (1 / group_seq_len_deviation + 1e-3))
        # print(weights)
        sampled_indexes_idx = torch.multinomial(weights, experience_count_n_samples, replacement=False).tolist()
        # print(sampled_indexes_idx)
        sampled_indexes = group_ready_for_consume_idx[sampled_indexes_idx].flatten().tolist()
    else:
        # weights = torch.sigmoid(1 / (torch.abs(seq_len - target_seq_len) + 1e-3))
        weights = 1 - torch.exp(-( 1 / torch.abs(seq_len - target_seq_len) + 1e-3))
        # print(weights)
        sampled_indexes_idx = torch.multinomial(weights, experience_count, replacement=False).tolist()
        # print(sampled_indexes_idx)
        sampled_indexes = [int(ready_for_consume_idx[i]) for i in sampled_indexes_idx]

    return sampled_indexes


def dp_token_load_balancing_strategy(
        ready_for_consume_idx: List[int], experience_count: int, get_n_samples: bool, n_samples_per_prompt: int,
        seq_len: Tensor) -> Optional[List[int]]:
    """
    get index using karmarkar_karp strategy across DP to make sure:
        1. each DP gets the same number of seqs and close total seq_len
        2. the standard deviation of total seq len across all DPs is small
    need to pass seq_len_list, which is the corresponding List of seq len for ready_for_consume_idx
    这个装箱算法只能保证DP token数一样
    """
    assert len(ready_for_consume_idx) == len(seq_len)

    if len(ready_for_consume_idx) == experience_count:
        sampled_indexes = [int(ready_for_consume_idx[i]) for i in range(experience_count)]
        return sampled_indexes

    assert len(ready_for_consume_idx) % experience_count == 0

    if get_n_samples:
        assert len(ready_for_consume_idx) % n_samples_per_prompt == 0
        assert experience_count % n_samples_per_prompt == 0
        experience_count_n_samples = experience_count // n_samples_per_prompt
        group_ready_for_consume_idx = torch.tensor(ready_for_consume_idx,
                                                   dtype=torch.int).view(-1, n_samples_per_prompt)
        group_seq_len_sum = seq_len.view(-1, n_samples_per_prompt).sum(dim=1)
        k_partitions = len(group_seq_len_sum) // experience_count_n_samples
        sampled_indexes_idx = get_seqlen_balanced_partitions(group_seq_len_sum.tolist(), k_partitions, equal_size=True)
        if len(sampled_indexes_idx) > 0:
            sampled_indexes = sum([group_ready_for_consume_idx[i].flatten().tolist() for i in sampled_indexes_idx[0]], [])
        else:
            sampled_indexes = None
    else:
        k_partitions = len(seq_len) // experience_count
        sampled_indexes_idx = get_seqlen_balanced_partitions(seq_len.tolist(), k_partitions, equal_size=True)
        # print(f'{sampled_indexes_idx=}')
        if len(sampled_indexes_idx) > 0:
            sampled_indexes = [int(ready_for_consume_idx[i]) for i in sampled_indexes_idx[0]]
        else:
            sampled_indexes = None
    return sampled_indexes

def similar_seq_len_strategy(
        ready_for_consume_idx: List[int], experience_count: int, get_n_samples: bool, n_samples_per_prompt: int,
        seq_len: Tensor) -> Optional[List[int]]:
    """
    cluster by seq_len and batch the groups/seqs with similar length together
    """
    pass

def storage_unit_load_balancing_strategy(
        ready_for_consume_idx: List[int], experience_count: int, get_n_samples: bool, storage_node_num: int, *args, **kwargs
) -> Optional[List[int]]:
    """
    sampling using round robin strategy across all storage nodes
    """
    pass
    # raise NotImplementedError


## ============ utils for samplers ============ ##

def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """Karmarkar-Karp algorithm for partitioning a list of integers into k partitions
    such that the difference between the largest and smallest partition is minimized.
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            number of partitions
        equal_size (bool):
            whether to make partitions equal size
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """

    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for s in self.sets:
                cur_partition = []
                for idx, _ in s.items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        if len(seqlen_list) % k_partitions != 0:
            raise ValueError(f"{len(seqlen_list)} % {k_partitions} != 0")
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for partition in partitions:
            if len(partition) * k_partitions != len(seqlen_list):
                raise ValueError(f"{len(partition)} * {k_partitions} != {len(seqlen_list)}")
    return partitions


def heapq_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    equal_part_num = len(seqlen_list) // k_partitions

    sorted_seqlen = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)

    # Initialize the heap: each group maintains [current sum, number of elements, group index, elements in the group]
    groups = [[0, 0, i, []] for i in range(k_partitions)]
    heapq.heapify(groups)

    partitions = []
    for seqlen, i in sorted_seqlen:
        current_group = heapq.heappop(groups)
        current_group[3].append(i)
        current_group[0] += seqlen
        current_group[1] += 1
        if equal_size:
            if current_group[1] < equal_part_num:
                heapq.heappush(groups, current_group)
            else:
                partitions.append(current_group[3])
        else:
            heapq.heappush(groups, current_group)

    partitions.extend([group[3] for group in groups])

    if equal_size:
        for i, partition in enumerate(partitions):
            if len(partition) * k_partitions != len(seqlen_list):
                raise ValueError(
                    f"Partition {i} has {len(partition)} items, expected {len(seqlen_list) // k_partitions}")
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seq length across dp ranks and micro batches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    if k_partitions > len(seqlen_list):
        raise ValueError(f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]")

    def _check_and_sort_partitions(partitions):
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        return sorted_partitions

    partitions = heapq_partition(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def rearrange_micro_batches(seqlen_list: List[int], max_token_len: int, dynamic_max_batch_size=None, dp_group=None):
    """get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seq length across dp ranks and micro batches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        max_token_len (int):
     Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    if max(seqlen_list) > max_token_len:
        raise ValueError(f"seqlen of items:[{max(seqlen_list)}] must <= max_token_len:[{max_token_len}]")

    # Calculate the minimum number of bins
    total_sum_of_seqlen = sum(seqlen_list)
    k_partitions = (total_sum_of_seqlen + max_token_len - 1) // max_token_len

    if dynamic_max_batch_size is not None:
        k_partitions = max(k_partitions, (len(seqlen_list) + dynamic_max_batch_size - 1) // dynamic_max_batch_size)

    if dist.is_initialized():
        k_partitions = torch.tensor([k_partitions], device='npu')
        dist.all_reduce(k_partitions, op=dist.ReduceOp.MAX, group=dp_group)
        k_partitions = k_partitions.cpu().item()

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=False)

    return partitions


def get_reverse_idx(idx_map):
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map