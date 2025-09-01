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
        self.meta = None

        # 1) 读 config.json，生成 meta
        cfg_path = self.model_path / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        hs   = int(cfg.get("hidden_size", 1536))
        nl   = int(cfg.get("num_hidden_layers", 28))
        nh   = int(cfg.get("num_attention_heads", 12))
        nkvh = int(cfg.get("num_key_value_heads", nh))
        di   = int(cfg.get("intermediate_size", 4 * hs))
        voc  = int(cfg.get("vocab_size", 151936))
        eps  = float(cfg.get("rms_norm_eps", 1e-6))
        theta= float(cfg.get("rope_theta", 10000.0))
        maxs = int(cfg.get("max_position_embeddings", 4096))
        eos  = int(cfg.get("eos_token_id", -1))

        dh = hs // nh

        meta = LlaisysQwen2Meta()
        meta.dtype     = DataType.F32
        meta.nlayer    = nl
        meta.hs        = hs
        meta.nh        = nh
        meta.nkvh      = nkvh
        meta.dh        = dh
        meta.di        = di
        meta.maxseq    = maxs
        meta.voc       = voc
        meta.epsilon   = eps
        meta.theta     = theta
        meta.end_token = eos
        self.meta = meta

     
        # 2) 创建 C++ 模型
        self.model = C.llaisysQwen2ModelCreate(meta, device, None, 0)
        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model.")
        # ⚠️ 这里保留的是 pointer，而不是 struct 拷贝
        self.weights = C.llaisysQwen2ModelWeights(self.model)

        # 3) 加载 safetensors
        for file in sorted(self.model_path.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tt = f.get_tensor(name)
                    arr = _to_f32_numpy(tt)
                    t = _make_tensor_from_numpy(arr, device)
                    self._assign_weight_by_name(name, t)

    def _assign_weight_by_name(self, name: str, t):
        w = self.weights.contents
        nl = self.meta.nlayer



        if name == "model.embed_tokens.weight":
            w.in_embed = t;  return
        if name == "lm_head.weight":
            w.out_embed = t;  return
        if name == "model.norm.weight":
            w.out_norm_w = t;  return

        prefix = "model.layers."
        if not name.startswith(prefix):
            print(f"  [SKIP] not a layer param"); return

        rest = name[len(prefix):]
        try:
            i_str, tail = rest.split(".", 1)
            i = int(i_str)
        except Exception:
            print(f"  [SKIP] bad name: {rest}"); return

        if i < 0 or i >= nl:
            print(f"  [SKIP] layer {i} out of range (nlayer={nl})")
            return

        # 根据 tail 分配
        if tail == "input_layernorm.weight":
            w.attn_norm_w[i] = t;  return
        if tail == "post_attention_layernorm.weight":
            w.mlp_norm_w[i] = t;  return

        if tail == "self_attn.q_proj.weight":
            w.attn_q_w[i] = t;  return
        if tail == "self_attn.q_proj.bias":
            w.attn_q_b[i] = t;  return
        if tail == "self_attn.k_proj.weight":
            w.attn_k_w[i] = t;  return
        if tail == "self_attn.k_proj.bias":
            w.attn_k_b[i] = t;  return
        if tail == "self_attn.v_proj.weight":
            w.attn_v_w[i] = t;  return
        if tail == "self_attn.v_proj.bias":
            w.attn_v_b[i] = t;  return
        if tail == "self_attn.o_proj.weight":
            w.attn_o_w[i] = t;  return

        if tail == "mlp.gate_proj.weight":
            w.mlp_gate_w[i] = t;  return
        if tail == "mlp.up_proj.weight":
            w.mlp_up_w[i] = t;  return
        if tail == "mlp.down_proj.weight":
            w.mlp_down_w[i] = t;  return

        print(f"  [UNUSED] {tail}")


    # ---- 生成（沿用 C++ 贪心/占位的输出） -------------------------------

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        verbose: bool = False,
    ):
        """
        简单的贪心/采样解码逻辑
        - 会在遇到 eos_token_id 或达到 max_new_tokens 时停止
        - 如果两者都没有，就默认最多跑 512 步避免死循环
        """

        # 1) eos
        eos_id = self.meta.end_token
        if eos_id is None:
            print("[generate] eos token setting wrong, stop.")
            return out
   

        # 2) prefill
        ids = np.asarray(list(inputs), dtype=np.int64)
        ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        nxt = C.llaisysQwen2ModelInfer(self.model, ids_ptr, ctypes.c_size_t(ids.size))

        out: list[int] = list(inputs)
        new_tokens = 0

        if nxt is None or nxt < 0:
            print("[generate] prefill returned None/<0>, stop.")
            return out

        token = int(nxt)
        out.append(token)
        last = token
        new_tokens += 1

        if eos_id is not None and last == eos_id:
            if verbose:
                print("[generate] hit EOS on prefill, stop.")
            return out

        # 3) decode loop
        limit = max_new_tokens if max_new_tokens is not None else 512

        while new_tokens < limit:
            nxt = C.llaisysQwen2ModelForwardOne(self.model, ctypes.c_int64(last))
            if nxt is None or nxt < 0:           # 先判空/判负
                if verbose:
                    print("[generate] forward_one returned None/<0>, stop.")
                return out

            token = int(nxt)
            out.append(token)
            last = token
            new_tokens += 1

            if eos_id is not None and token == eos_id:
                if verbose:
                    print("[generate] hit EOS, stop.")
                return out

        print("[generate] reach context limit, stop.")
        return out


    def __del__(self):
        try:
            if getattr(self, "model", None):
                C.llaisysQwen2ModelDestroy(self.model)
                self.model = None
        except Exception:
            pass
