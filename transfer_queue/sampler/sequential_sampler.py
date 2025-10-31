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
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from transfer_queue.sampler import BaseSampler

class SequentialSampler(BaseSampler):
    def __init__(self,):
        super().__init__()

    def sample(
            self,
            ready_indexes: list[int],
            batch_size: int,
    ) -> tuple[list[int], list[int]]:
        sampled_indexes = ready_indexes[:batch_size]
        consumed_indexes = sampled_indexes

        return sampled_indexes, consumed_indexes