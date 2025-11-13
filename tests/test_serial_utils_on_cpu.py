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

import sys
from pathlib import Path

import pytest
import torch


# Import your classes here
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
def test_tensor_serialization(dtype):
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(torch.Tensor)

    tensor = torch.randn(100, 10, dtype=dtype)
    serialized = encoder.encode(tensor)
    deserialized = decoder.decode(serialized)
    assert torch.allclose(tensor, deserialized)

    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    input_ids = [a, b]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged, dtype=dtype)

    input_ids_serialized = encoder.encode(input_ids)
    input_ids_deserialized = decoder.decode(input_ids_serialized)
    for i in range(len(input_ids.unbind())):
        assert torch.allclose(input_ids[i], input_ids_deserialized[i])
