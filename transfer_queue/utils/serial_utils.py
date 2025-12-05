# Copyright 2025 The TransferQueue Team
# Copyright 2025 The vLLM project
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

# This implementation is inspired by https://github.com/vllm-project/vllm/blob/main/vllm/v1/serial_utils.py

import pickle
from collections.abc import Sequence
from inspect import isclass
from types import FunctionType
from typing import Any, Optional, TypeAlias

import cloudpickle
import torch
import zmq
from msgspec import msgpack

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

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: Optional[list[bytestr]] = None

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
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_tensor(self, obj: torch.Tensor) -> tuple[str, list[tensorenc]] | tensorenc:
        assert self.aux_buffers is not None
        # view the tensor as a contiguous 1D array of bytes
        arr = obj.flatten().contiguous().view(torch.uint8).numpy()
        data = len(self.aux_buffers)
        self.aux_buffers.append(arr.data)
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data


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
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
        return obj

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
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
        # Create uint8 array and convert read-only buffer into writable bytearray
        arr = torch.frombuffer(bytearray(buffer), dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")
