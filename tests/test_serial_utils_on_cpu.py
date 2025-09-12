import pytest
import torch
import tensordict
import numpy as np
from tensordict import NonTensorData, NonTensorStack, TensorDict
from transfer_queue.utils.serial_utils import MsgpackEncoder, MsgpackDecoder


def get_tensordict(tensor_dict: dict[str, torch.Tensor | list], non_tensor_dict: dict = None) -> TensorDict:
    if non_tensor_dict is None:
        non_tensor_dict = {}

    batch_size = None

    for key, val in tensor_dict.items():
        if isinstance(val, list):
            for v in val:
                assert not isinstance(v, torch.Tensor), (
                    "Passing a list makes the data NonTensorStack, "
                    "which doesn't support torch.Tensor. Please convert to numpy first"
                )

        assert isinstance(val, torch.Tensor | list)

        if batch_size is None:
            batch_size = len(val)
        else:
            assert len(val) == batch_size

    if batch_size is None:
        batch_size = []
    else:
        batch_size = [batch_size]

    for key, val in non_tensor_dict.items():
        assert key not in tensor_dict
        tensor_dict[key] = NonTensorData(val)

    return TensorDict(source=tensor_dict, batch_size=batch_size)


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
        assert torch.allclose(input_ids[0], input_ids_deserialized[0])


def test_tensordict_serialization_with_nontensor():
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(TensorDict)

    obs = torch.randn(100, 10)
    data1 = {"obs": obs, "act": torch.randn(100, 3), "data_sources": ["gsm8k"] * 100}
    data1 = get_tensordict(tensor_dict=data1)

    serialized = encoder.encode(data1)
    deserialized = decoder.decode(serialized)

    assert deserialized.keys() == data1.keys()
    assert deserialized.batch_size[0] == 100
    assert isinstance(deserialized.get("data_sources"), NonTensorStack)
    for k, v in data1.items():
        if isinstance(v, torch.Tensor):
            assert torch.allclose(deserialized[k], v)
        elif isinstance(v, NonTensorStack):
            assert deserialized[k] == data1[k]


def test_tensordict_serialization_with_images():
    # each sample contains a sequence with multiple images of different sizes
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    input_ids = [a, b]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)

    a_images = [
        torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8).numpy(),
        torch.randint(low=0, high=255, size=(3, 128, 128), dtype=torch.uint8).numpy(),
    ]
    b_images = [
        torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8).numpy(),
        torch.randint(low=0, high=255, size=(3, 128, 128), dtype=torch.uint8).numpy(),
        torch.randint(low=0, high=255, size=(3, 64, 64), dtype=torch.uint8).numpy(),
    ]

    images = [a_images, b_images]

    data = get_tensordict({"input_ids": input_ids, "images": images})

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(TensorDict)

    serialized = encoder.encode(data)
    deserialized = decoder.decode(serialized)

    assert np.all(np.equal(deserialized[0]["images"][0], a_images[0]))
    assert torch.all(torch.eq(deserialized[0]["input_ids"], a))


# Copied from https://github.com/volcengine/verl/blob/33edd95e13c72b9494585765b5fedc679fd73923/tests/test_protocol_v2_on_cpu.py#L119
def test_tensordict_with_packing():
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    input_ids = [a, b]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)

    data = get_tensordict({"input_ids": input_ids})
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(TensorDict)
    deserialized_data = decoder.decode(encoder.encode(data))

    # test cu_seqlens
    cu_seqlens = torch.tensor([0, 11, 24])
    assert torch.all(torch.eq(cu_seqlens, deserialized_data["input_ids"].offsets()))

    # test index
    assert torch.all(torch.eq(deserialized_data["input_ids"][0], a))
    assert torch.all(torch.eq(deserialized_data["input_ids"][1], b))

    assert torch.all(torch.eq(deserialized_data[0]["input_ids"], a))
    assert torch.all(torch.eq(deserialized_data[1]["input_ids"], b))

    data_lst = deserialized_data.chunk(2)

    assert torch.all(torch.eq(data_lst[0]["input_ids"][0], a))
    assert torch.all(torch.eq(data_lst[1]["input_ids"][0], b))


def test_nested_tensordict_serialization():
    td1 = tensordict.TensorDict({
        'a': torch.randn(2, 3),
        'b': torch.randn(2, 4)
    }, batch_size=[2])

    td2 = tensordict.TensorDict({
        'c': torch.randn(2, 5),
        'd': torch.randn(2, 6)
    }, batch_size=[2])

    td = tensordict.TensorDict({
        'part1': td1,
        'part2': td2,
        'e': torch.randn(2, 7)
    }, batch_size=[2])

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(TensorDict)
    deserialized_td = decoder.decode(encoder.encode(td))

    assert isinstance(deserialized_td, tensordict.TensorDict)
    assert set(deserialized_td.keys()) == set(td.keys())
    assert isinstance(deserialized_td['part1'], tensordict.TensorDict)
    assert isinstance(deserialized_td['part2'], tensordict.TensorDict)

    assert set(deserialized_td['part1'].keys()) == set(td1.keys())
    assert set(deserialized_td['part2'].keys()) == set(td2.keys())

    for key in td.keys():
        if isinstance(td[key], tensordict.TensorDict):
            for inner_key in td[key].keys():
                assert torch.allclose(deserialized_td[key][inner_key], td[key][inner_key]), \
                    f"Values for key '{key}.{inner_key}' do not match"
        else:
            assert torch.allclose(deserialized_td[key], td[key]), f"Values for key '{key}' do not match"
