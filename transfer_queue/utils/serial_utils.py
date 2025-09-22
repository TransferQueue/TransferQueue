import os
import pickle
from collections.abc import Sequence
from inspect import isclass
from types import FunctionType
from typing import Any, Optional, TypeAlias

import cloudpickle
import torch
import zmq
from msgspec import msgpack
from tensordict import NonTensorData, TensorDict

TQ_MSGPACK_ZERO_COPY_THRESHOLD = int(os.environ.get("TQ_MSGPACK_ZERO_COPY_THRESHOLD", 256))
CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame
tensorenc = tuple[str, tuple[int, ...], int | memoryview]


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.
    """

    def __init__(self, size_threshold: Optional[int] = None):
        if size_threshold is None:
            size_threshold = TQ_MSGPACK_ZERO_COPY_THRESHOLD
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: Optional[list[bytestr]] = None
        self.size_threshold = size_threshold

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, TensorDict):
            return self._encode_tensordict(obj)

        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_tensordict(self, obj: TensorDict) -> tuple[tuple[int, ...], Optional[str], dict[str, tuple[str, Any]]]:
        assert self.aux_buffers is not None
        encoded_items: dict[str, tuple[str, Any]] = {}
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                encoded_items[k] = ("tensor", self._encode_tensor(v))
            # elif isinstance(v, NonTensorStack):
            #     encoded_items[k] = ("non_tensor_stack", self._encode_non_tensor_stack(v))
            elif isinstance(v, NonTensorData):
                encoded_items[k] = ("non_tensor_data", self._encode_non_tensor_data(v))
            else:
                data = len(self.aux_buffers)
                self.aux_buffers.append(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL))
                encoded_items[k] = ("other", data)
        batch_size = tuple(obj.batch_size)
        device = str(obj.device) if obj.device is not None else None
        return batch_size, device, encoded_items

    def _encode_tensor(self, obj: torch.Tensor) -> tuple[str, list[tensorenc]] | tensorenc:
        if not obj.is_nested:
            return self._encode_single_tensor(obj)
        else:
            layout = str(obj.layout).removeprefix("torch.")
            data = [self._encode_single_tensor(tensor) for tensor in obj.unbind()]
            return layout, data

    def _encode_single_tensor(self, obj: torch.Tensor) -> tensorenc:
        assert self.aux_buffers is not None
        # view the tensor as a contiguous 1D array of bytes
        arr = obj.flatten().contiguous().view(torch.uint8).numpy()
        if obj.nbytes < self.size_threshold:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr.data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr.data)
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    def _encode_non_tensor_data(self, obj: NonTensorData) -> tuple[tuple[int, ...], Optional[str], int]:
        assert self.aux_buffers is not None
        batch_size = tuple(obj.batch_size)
        device = str(obj.device) if obj.device is not None else None
        data = len(self.aux_buffers)
        self.aux_buffers.append(pickle.dumps(obj.data, protocol=pickle.HIGHEST_PROTOCOL))
        return batch_size, device, data


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Optional[Any] = None):
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(*args, ext_hook=self.ext_hook, dec_hook=self.dec_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])  # type: ignore[index]
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, TensorDict):
                return self._decode_tensordict(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
        return obj

    def _decode_tensordict(self, arr: Any) -> TensorDict:
        batch_size, device, encoded_items = arr
        decoded_items: dict[str, Any] = {}

        for k, (v_type, v) in encoded_items.items():
            if v_type == "tensor":
                decoded_items[k] = self._decode_tensor(v)
            # elif v_type == "non_tensor_stack":
            #     decoded_items[k] = self._decode_non_tensor_stack(v)
            elif v_type == "non_tensor_data":
                decoded_items[k] = self._decode_non_tensor_data(v)
            elif v_type == "other":
                decoded_items[k] = pickle.loads(self.aux_buffers[v])

        batch_size = torch.Size(batch_size)
        torch_device = torch.device(device) if device is not None else None

        return TensorDict(source=decoded_items, batch_size=batch_size, device=torch_device)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        if len(arr) == 3:
            # decode single tensor
            return self._decode_single_tensor(arr)
        elif len(arr) == 2:
            # decode nested tensor
            layout, data = arr
            torch_layout = getattr(torch, layout)
            return torch.nested.as_nested_tensor(
                [self._decode_single_tensor(tensor) for tensor in data], layout=torch_layout
            )
        else:
            raise ValueError(f"Invalid tensor encoding format, expected length 2 or 3, got {len(arr)}")

    def _decode_single_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        # Copy from inline representation, to decouple the memory storage
        # of the message from the original buffer. And also make Torch
        # not complain about a readonly memoryview.
        buffer = self.aux_buffers[data] if isinstance(data, int) else bytearray(data)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def _decode_non_tensor_data(self, arr: Any) -> NonTensorData:
        batch_size, device, data = arr
        buffer = self.aux_buffers[data]
        batch_size = torch.Size(batch_size)
        torch_device = torch.device(device) if device is not None else None
        non_tensor_data = pickle.loads(buffer)
        return NonTensorData(data=non_tensor_data, batch_size=batch_size, device=torch_device)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")
