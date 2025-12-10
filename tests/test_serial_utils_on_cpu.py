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
from tensordict import TensorDict

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


def test_zmq_msg_serialization():
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # construct complex msg body with nested tensor, jagged tensor, normal tensor, numpy array
    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test_sender",
        receiver_id="test_receiver",
        request_id="test_request",
        timestamp="test_timestamp",
        body={
            "data": TensorDict(
                {
                    "nested_tensor": torch.nested.as_nested_tensor(
                        [torch.randn(4, 3), torch.randn(2, 4)], layout=torch.strided
                    ),
                    "jagged_tensor": torch.nested.as_nested_tensor(
                        [torch.randn(4, 5), torch.randn(4, 54)], layout=torch.jagged
                    ),
                    "normal_tensor": torch.randn(2, 10, 3),
                    "numpy_array": torch.randn(2, 2).numpy(),
                },
                batch_size=2,
            )
        },
    )
    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)
    assert decoded_msg.request_type == msg.request_type
    assert torch.allclose(decoded_msg.body["data"]["numpy_array"], msg.body["data"]["numpy_array"])
    assert torch.allclose(decoded_msg.body["data"]["normal_tensor"], msg.body["data"]["normal_tensor"])
    for i in range(len(msg.body["data"]["nested_tensor"].unbind())):
        assert torch.allclose(
            decoded_msg.body["data"]["nested_tensor"][i],
            msg.body["data"]["nested_tensor"][i],
        )
    for i in range(len(msg.body["data"]["jagged_tensor"].unbind())):
        assert torch.allclose(
            decoded_msg.body["data"]["jagged_tensor"][i],
            msg.body["data"]["jagged_tensor"][i],
        )


@pytest.mark.parametrize(
    "make_view",
    [
        lambda x: x[:, :5],  # 前半列
        lambda x: x[::2],  # 跨步取行
        lambda x: x[..., 1:],  # 去掉第一个列
        lambda x: x.transpose(0, 1),  # 维度交换（通常非 contiguous）
        lambda x: x[1:-1, 2:8:2],  # 组合切片
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
def test_tensor_serialization_with_views(dtype, make_view):
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(torch.Tensor)

    base = torch.randn(16, 16, dtype=dtype)
    view = make_view(base)

    print("is_view_like:", view._base is not None, "is_contiguous:", view.is_contiguous())

    serialized = encoder.encode(view)
    deserialized = decoder.decode(serialized)

    assert deserialized.shape == view.shape
    assert torch.allclose(view, deserialized)
