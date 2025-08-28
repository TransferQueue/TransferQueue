import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()

#### copied from simple_sampling_test.py
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# import numpy as np
# import torch
# from torch import Tensor
# from typing import List, Dict, Optional, Union, Any, Callable, TypedDict
# from load_balance_strategy import random_strategy, dp_token_load_balancing_strategy, similar_seq_len_strategy, \
#     target_seq_len_strategy, storage_unit_load_balancing_strategy
#
# import random
#
# # 设置 PyTorch 的随机种子
# def set_torch_seed(seed):
#     random.seed(seed)           # Python 随机数生成器
#     np.random.seed(seed)        # NumPy 随机数生成器
#     torch.manual_seed(seed)     # PyTorch CPU 随机种子
#     torch.cuda.manual_seed(seed) # PyTorch GPU 随机种子
#     torch.cuda.manual_seed_all(seed) # 如果使用多个GPU
#
#     # 添加以下设置以提高可重现性（可能会降低性能）
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#     # 设置环境变量
#     import os
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
# # 使用函数设置种子
# set_torch_seed(42)
#
# def test():
#     schedule_policies = {"random_strategy": random_strategy,
#                          "dp_token_load_balancing_strategy": dp_token_load_balancing_strategy,
#                          "target_seq_len_strategy": target_seq_len_strategy,
#                          "similar_seq_len_strategy": similar_seq_len_strategy,
#                          "storage_unit_load_balancing_strategy": storage_unit_load_balancing_strategy,
#                         }
#
#     experience_count = 8 # 2*4
#
#     ready_for_consume_idx = [0, 1, 2, 3,
#                              4, 5, 6, 7,
#                              8, 9, 10, 11,
#                              12, 13, 14, 15,
#                              16, 17, 18, 19,
#                              24, 25, 26, 27]
#
#
#     seq_len_list = [48, 49, 50, 51,  # 198
#                     75, 50, 75, 100,  # 300
#                     45, 55, 40, 60,  # 200
#                     45, 55, 40, 60,  # 200
#                     100, 25, 50, 125,  # 300
#                     75, 50, 100, 75]  # 300
#
#     seq_len = torch.tensor(seq_len_list, dtype=torch.int)
#
#     target_seq_len = 50
#
#     for policy_name in schedule_policies.keys():
#         for get_n_samples in [False, True]:
#             if get_n_samples:
#                 num_n_samples = 4
#             else:
#                 num_n_samples = 1
#             if policy_name == "random_strategy":
#                 sampled_indexes = run_schedule_policy(schedule_policies, policy_name, ready_for_consume_idx, experience_count, get_n_samples,
#                                     num_n_samples)
#             elif policy_name == "dp_token_load_balancing_strategy":
#                 sampled_indexes = run_schedule_policy(schedule_policies, policy_name, ready_for_consume_idx, experience_count, get_n_samples,
#                                     num_n_samples, seq_len)
#             elif policy_name == "target_seq_len_strategy":
#                 sampled_indexes = run_schedule_policy(schedule_policies, policy_name, ready_for_consume_idx, experience_count, get_n_samples,
#                                     num_n_samples, seq_len, target_seq_len)
#
#             # elif policy_name == "similar_seq_len_strategy":
#             #     sampled_indexes = None
#             #     pass  ## not implemented yet
#
#             # elif policy_name == "storage_unit_load_balancing_strategy":
#             #     sampled_indexes = None
#             #     pass  ## not implemented yet
#
#             else:
#                 sampled_indexes = None
#                 pass
#             print(f'under policy {policy_name} and get_n_samples:{get_n_samples}, {sampled_indexes=}')
#
#
#
#
# def run_schedule_policy(schedule_policies: dict, policy_name: str, ready_for_consume_idx: List[int], experience_count: int,
#                         get_n_samples: bool = False, num_n_samples:int=1, *args, **kwargs) -> Optional[List[int]]:
#     """
#     run the scheduler policy based on policy_name, ready_for_consume_idx, required experience_count
#     now the schedule process is called by trainer and sample indexes for all DPs are decided altogether
#     policy should not be bothered by get_n_samples requirement
#     example:
#         if get_n_samples:
#             ready_for_consume_idx could look like: [0, 1, 2, 3,   8, 9, 10, 11,   16, 17, 18, 19]
#         else:
#             ready_for_consume_idx could look like: [2, 5, 6]
#     """
#     if len(ready_for_consume_idx) < experience_count:
#         # logger.info('Error: not enough data to consume yet.')
#         return None
#
#     return schedule_policies[policy_name](ready_for_consume_idx,
#                                           experience_count,
#                                           get_n_samples,
#                                           num_n_samples,
#                                           *args, **kwargs)
#
#     # # 准备传递给策略函数的参数
#     # policy_args = {
#     #     'ready_for_consume_idx': ready_for_consume_idx,
#     #     'experience_count': experience_count,
#     #     'get_n_samples': get_n_samples,
#     #     'n_samples_per_prompt': num_n_samples,
#     # }
#     #
#     # # 添加其他可能需要的参数
#     # policy_args.update(kwargs)
#     #
#     # # 调用策略函数
#     # return schedule_policies[policy_name](**policy_args)
#
# test()