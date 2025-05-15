import os
import inspect
import contextlib
import numpy as np

from collections import OrderedDict

import mindspore
from mindspore import context, nn, mutable, ParallelMode
from mindspore.common.api import _pynative_executor


NAN_TENSOR = mindspore.Tensor(np.array(float('nan')), dtype=mindspore.float32)


_DTYPE_2_STRING = {
    mindspore.float16: "float16",
    mindspore.bfloat16: "bfloat16",
    mindspore.float32: "float32",
    mindspore.float64: "float64",
    mindspore.uint8: "uint8",
    mindspore.int8: "int8",
    mindspore.int16: "int16",
    mindspore.int32: "int32",
    mindspore.int64: "int64",
    mindspore.bool_: "bool",
}

_STRING_2_DTYPE = {
    "float16": mindspore.float16,
    "bfloat16": mindspore.bfloat16,
    "float32": mindspore.float32,
    "float64": mindspore.float64,
    "uint8": mindspore.uint8,
    "int8": mindspore.int8,
    "int16": mindspore.int16,
    "int32": mindspore.int32,
    "int64": mindspore.int64,
    "bool": mindspore.bool_,
}


_MIN_FP16 = mindspore.tensor(np.finfo(np.float16).min, dtype=mindspore.float16)
_MIN_FP32 = mindspore.tensor(np.finfo(np.float32).min, dtype=mindspore.float32)
_MIN_FP64 = mindspore.tensor(np.finfo(np.float64).min, dtype=mindspore.float64)
_MIN_BF16 = mindspore.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=mindspore.bfloat16)
_MAX_FP16 = mindspore.tensor(np.finfo(np.float16).max, dtype=mindspore.float16)
_MAX_FP32 = mindspore.tensor(np.finfo(np.float32).max, dtype=mindspore.float32)
_MAX_FP64 = mindspore.tensor(np.finfo(np.float64).max, dtype=mindspore.float64)
_MAX_BF16 = mindspore.tensor(float.fromhex("0x1.fe00000000000p+127"), dtype=mindspore.bfloat16)


_DTYPE_2_MIN = {
    mindspore.float16: _MIN_FP16,
    mindspore.float32: _MIN_FP32,
    mindspore.float64: _MIN_FP64,
    mindspore.bfloat16: _MIN_BF16,
}

_DTYPE_2_MAX = {
    mindspore.float16: _MAX_FP16,
    mindspore.float32: _MAX_FP32,
    mindspore.float64: _MAX_FP64,
    mindspore.bfloat16: _MAX_BF16,
}


def dtype_to_min(dtype):
    return _DTYPE_2_MIN.get(dtype, "others dtype")


def dtype_to_max(dtype):
    return _DTYPE_2_MAX.get(dtype, "others dtype")


def dtype_to_str(dtype):
    return _DTYPE_2_STRING.get(dtype, "others dtype")


def str_to_dtype(dtype):
    return _STRING_2_DTYPE.get(dtype, "others dtype")


def _is_parallel():
    return mindspore.context.get_auto_parallel_context("parallel_mode") not in (ParallelMode.STAND_ALONE,)


def _is_graph():
    return mindspore.context.get_context("mode") == mindspore.GRAPH_MODE


def _is_ascend():
    return mindspore.context.get_context("device_target") == "Ascend"


# FIXME: Can't work on MindSpore 2.3.0
# @mindspore.constexpr(reuse_result=False)
# def _tensor_2_tuple(input):
#     return tuple(input.asnumpy().tolist())


@mindspore.jit_class
class pynative_no_grad(contextlib.ContextDecorator):
    """
    Context Manager to disable gradient calculation. When enter this context, we will disable calculate
    gradient. When exit this context, we will resume its prev state.
    Currently, it can use both in Pynative and Graph mode. It also can be used as decorator.

    For mindone.diffusers, it is used in PyNative training to decorate the part of calculation that
    does not require gradients, e.g. vae.encode_images or text_encoder.encode_prompts where does not
    need to train VAE or text-encoders.
    """

    def __init__(self):
        self.is_pynative_mode = context.get_context("mode") == context.PYNATIVE_MODE or os.getenv("MS_JIT") == "0"
        self.prev_state = False

    def __enter__(self):
        if self.is_pynative_mode:
            self.prev_state = _pynative_executor.enable_grad()
            _pynative_executor.set_enable_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_pynative_mode:
            _pynative_executor.set_enable_grad(self.prev_state)
        return False


def get_tensor_dynamic_input(tensors):
    if tensors is None:
        return None
    elif isinstance(tensors, mindspore.Tensor):
        if -1 in tensors.shape:
            return mindspore.Tensor(shape=[(None if i == -1 else i) for i in tensors.shape], dtype=tensors.dtype)
        else:
            return mindspore.Tensor(shape=[None for _ in range(tensors.ndim)], dtype=tensors.dtype)
    elif isinstance(tensors, (list, tuple)):
        return mutable([get_tensor_dynamic_input(t) for t in tensors])
    elif isinstance(tensors, (int, float)):
        return tensors
    elif isinstance(tensors, bool):
        return tensors
    else:
        raise ValueError(f"enable_dynamic_shape: got unexpected types of data, current data: {tensors}")


def enable_dynamic_shape(cell: nn.Cell, *cell_inputs, **kwargs):

    assert isinstance(cell, nn.Cell)
    assert len(kwargs) == 0, "not support dict inputs"

    dynamic_inputs = []
    for input in cell_inputs:
        dynamic_input = get_tensor_dynamic_input(input)
        dynamic_inputs.append(dynamic_input)

    cell.set_inputs(*dynamic_inputs)
