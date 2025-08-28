from typing import Sequence
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

from ..libllaisys import LIB_LLAISYS as C
from ..libllaisys import DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from ..libllaisys.llaisys_types import DataType
import ctypes

def _make_tensor_from_numpy(arr: np.ndarray, device: DeviceType):
    # ---- dtype 统一到 float32 ----
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    ndim = arr.ndim
    shape_ctypes = (ctypes.c_size_t * ndim)(*arr.shape)

  
    dtype = DataType.F32

 
    t = C.tensorCreate(shape_ctypes, ctypes.c_size_t(ndim), dtype, device, 0)

    # void*
    C.tensorLoad(t, ctypes.c_void_p(arr.ctypes.data))
    return t


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.weights = None

        # 占位
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.F32
        meta.nlayer = 2
        meta.hs = 16
        meta.nh = 2
        meta.nkvh = 2
        meta.dh = 8
        meta.di = 32
        meta.maxseq = 16
        meta.voc = 100
        meta.epsilon = 1e-6
        meta.theta = 10000.0
        meta.end_token = -1

        # 创建 C++ 模型
        self.model = C.llaisysQwen2ModelCreate(meta, device, None, 0)
        self.weights = C.llaisysQwen2ModelWeights(self.model).contents

        # —— 遍历 safetensors 并加载（用 torch 读，转 float32）——
        for file in sorted(self.model_path.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tt = f.get_tensor(name)  # torch.Tensor on CPU
                    # 转 float32 + contiguous
                    if tt.dtype != torch.float32:
                        tt = tt.to(dtype=torch.float32)
                    tt = tt.contiguous()
                    arr = tt.numpy()  # 得到 numpy float32 (C 连续)

                    # 调试输出
                    print(f"[Qwen2] loading tensor: {name}, shape={tuple(arr.shape)}, dtype=float32")

                    t = _make_tensor_from_numpy(arr, device)

                    # 最小实现：先都塞进 in_embed 作占位
                    self.weights.in_embed = t

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
  
        ids = np.asarray(list(inputs), dtype=np.int64)

        #  转成 POINTER(c_int64) + c_size_t
        ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        nxt = C.llaisysQwen2ModelInfer(self.model, ids_ptr, ctypes.c_size_t(ids.size))

        out = list(inputs)
        if nxt >= 0:
            out.append(int(nxt))
            last = int(nxt)
        else:
            last = out[-1] if out else 0

        steps = (max_new_tokens or 0) - 1 if max_new_tokens else 0
        for _ in range(steps):
            #  单个 token 也包成 c_int64
            nxt = C.llaisysQwen2ModelForwardOne(self.model, ctypes.c_int64(last))
            if nxt < 0:
                break
            out.append(int(nxt))
            last = int(nxt)

        return out

    def __del__(self):
        try:
            if getattr(self, "model", None):
                C.llaisysQwen2ModelDestroy(self.model)
                self.model = None
        except Exception:
            pass
