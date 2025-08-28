from typing import Sequence
from pathlib import Path
import json
import ctypes
import numpy as np
import torch
from safetensors import safe_open

from ..libllaisys import LIB_LLAISYS as C
from ..libllaisys import DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from ..libllaisys.llaisys_types import DataType


# ---- helpers ---------------------------------------------------------------

def _make_tensor_from_numpy(arr: np.ndarray, device: DeviceType):
    """
    把 numpy 数组变成后端的 llaisysTensor_t（统一 float32，C 连续）
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    ndim = arr.ndim
    shape_ctypes = (ctypes.c_size_t * ndim)(*arr.shape)
    dtype = DataType.F32

    t = C.tensorCreate(shape_ctypes, ctypes.c_size_t(ndim), dtype, device, 0)
    C.tensorLoad(t, ctypes.c_void_p(arr.ctypes.data))
    return t


def _to_f32_numpy(t: torch.Tensor) -> np.ndarray:
    """
    torch.Tensor(CPU) -> float32 contiguous numpy.ndarray
    """
    if t.dtype != torch.float32:
        t = t.to(dtype=torch.float32)
    t = t.contiguous()
    return t.numpy()


def _get(d: dict, key: str, default=None):
    return d[key] if key in d and d[key] is not None else default


# ---- model -----------------------------------------------------------------

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.weights = None

        # 1) 读取 config.json，填 meta
        cfg_path = self.model_path / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        hs   = int(_get(cfg, "hidden_size",            1536))
        nl   = int(_get(cfg, "num_hidden_layers",      28))
        nh   = int(_get(cfg, "num_attention_heads",    12))
        nkvh = int(_get(cfg, "num_key_value_heads",    nh))
        di   = int(_get(cfg, "intermediate_size",      4 * hs))
        voc  = int(_get(cfg, "vocab_size",             151936))
        eps  = float(_get(cfg, "rms_norm_eps",         1.0e-6))
        theta= float(_get(cfg, "rope_theta",           10000.0))
        maxs = int(_get(cfg, "max_position_embeddings", 4096))
        eos  = int(_get(cfg, "eos_token_id",          -1))

        dh = hs // nh  # 每头维度

        meta = LlaisysQwen2Meta()
        meta.dtype      = DataType.F32
        meta.nlayer     = nl
        meta.hs         = hs
        meta.nh         = nh
        meta.nkvh       = nkvh
        meta.dh         = dh
        meta.di         = di
        meta.maxseq     = maxs
        meta.voc        = voc
        meta.epsilon    = eps
        meta.theta      = theta
        meta.end_token  = eos

        # 2) 创建 C++ 模型并获取权重槽
        self.model = C.llaisysQwen2ModelCreate(meta, device, None, 0)
        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model.")
        self.weights = C.llaisysQwen2ModelWeights(self.model).contents

        # 3) 加载 safetensors 权重并“按名装配”到 C++ 权重结构
        #    注意：此处只负责把数组搬到正确的槽位；真正算子执行在 C++ 里
        for file in sorted(self.model_path.glob("*.safetensors")):
            # 用 torch 框架读取（safe_open 会返回 torch.Tensor）
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tt = f.get_tensor(name)
                    arr = _to_f32_numpy(tt)

                    t = _make_tensor_from_numpy(arr, device)
                    self._assign_weight_by_name(name, t)

    # 把张量句柄按名称塞到对应的槽位里
    def _assign_weight_by_name(self, name: str, t):
        w = self.weights

        # top-level
        if name == "model.embed_tokens.weight":
            w.in_embed = t
            return
        if name == "lm_head.weight":
            w.out_embed = t
            return
        if name == "model.norm.weight":
            w.out_norm_w = t
            return

        # per-layer: "model.layers.{i}...."
        prefix = "model.layers."
        if not name.startswith(prefix):
            return

        rest = name[len(prefix):]  # e.g. "12.self_attn.q_proj.weight"
        layer_str, tail = rest.split(".", 1)
        try:
            i = int(layer_str)
        except ValueError:
            return  # 名字异常，忽略

        # attn norms
        if tail == "input_layernorm.weight":
            w.attn_norm_w[i] = t
            return

        # mlp norms
        if tail == "post_attention_layernorm.weight":
            w.mlp_norm_w[i] = t
            return

        # attention projections
        if tail == "self_attn.q_proj.weight":
            w.attn_q_w[i] = t
            return
        if tail == "self_attn.q_proj.bias":
            w.attn_q_b[i] = t
            return
        if tail == "self_attn.k_proj.weight":
            w.attn_k_w[i] = t
            return
        if tail == "self_attn.k_proj.bias":
            w.attn_k_b[i] = t
            return
        if tail == "self_attn.v_proj.weight":
            w.attn_v_w[i] = t
            return
        if tail == "self_attn.v_proj.bias":
            w.attn_v_b[i] = t
            return
        if tail == "self_attn.o_proj.weight":
            w.attn_o_w[i] = t
            return

        # MLP projections
        if tail == "mlp.gate_proj.weight":
            w.mlp_gate_w[i] = t
            return
        if tail == "mlp.up_proj.weight":
            w.mlp_up_w[i] = t
            return
        if tail == "mlp.down_proj.weight":
            w.mlp_down_w[i] = t
            return

        # 其它名暂不处理（例如 bias/rotary_emb 之类，不在最小实现中）
        return

    # ---- 生成（沿用 C++ 贪心/占位的输出） -------------------------------

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # prefill
        ids = np.asarray(list(inputs), dtype=np.int64)
        ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        nxt = C.llaisysQwen2ModelInfer(self.model, ids_ptr, ctypes.c_size_t(ids.size))

        out = list(inputs)
        if nxt >= 0:
            out.append(int(nxt))
            last = int(nxt)
        else:
            last = out[-1] if out else 0

        # decode loop
        steps = (max_new_tokens or 0) - 1 if max_new_tokens else 0
        for _ in range(steps):
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
