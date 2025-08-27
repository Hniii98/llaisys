from ctypes import (
    Structure, POINTER, c_size_t, c_float, c_int64, c_int
)
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


#


# Define structure 
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

# Define structure 
class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


class LlaisysQwen2Model(Structure):
    pass

llaisysQwen2Model_p = POINTER(LlaisysQwen2Model)


def load_qwen2(lib):
 
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta), # meta
        llaisysDeviceType_t, # device
        POINTER(c_int), # device_ids
        c_int # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_p

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_p]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_p,
        POINTER(c_int64),
        c_size_t
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelForwardOne.argtypes = [
        llaisysQwen2Model_p,
        c_int64
    ]
    lib.llaisysQwen2ModelForwardOne.restype = c_int64

    lib.llaisysQwen2ModelLogits.argtypes = [llaisysQwen2Model_p]
    lib.llaisysQwen2ModelLogits.restype = llaisysTensor_t


__all__ = [
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "llaisysQwen2Model_p",
    "llaisysQwen2ModelCreate",
    "llaisysQwen2ModelDestroy",
    "llaisysQwen2ModelWeights",
    "llaisysQwen2ModelInfer",
    "llaisysQwen2ModelForwardOne",
    "llaisysQwen2ModelLogits",
]