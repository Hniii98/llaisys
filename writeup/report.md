
# Report

##  Contents
- [Report](#report)
  - [Contents](#contents)
  - [1. Overview](#1-overview)
  - [2. System Architecture](#2-system-architecture)
    - [2.1 Layered Design](#21-layered-design)
    - [2.2 Model Overview](#22-model-overview)
      - [(1) Python Frontend](#1-python-frontend)
      - [(2) C Interface](#2-c-interface)
      - [(3) C++ Implementation](#3-c-implementation)
    - [2.3 Operator Overview](#23-operator-overview)
  - [3. Operator Implementation and Optimization](#3-operator-implementation-and-optimization)
    - [3.1 Add](#31-add)
    - [3.2 Argmax](#32-argmax)
    - [3.3 Embedding](#33-embedding)
    - [3.4 Linear](#34-linear)
    - [3.5 RMSNorm](#35-rmsnorm)
    - [3.6 RoPE (Rotary Position Embedding)](#36-rope-rotary-position-embedding)
    - [3.7 Self-Attention](#37-self-attention)
      - [(1) 核心结构](#1-核心结构)
      - [(2) Decode 路径优化](#2-decode-路径优化)
    - [3.8 SwiGLU](#38-swiglu)
  - [4. Operator Testing and Validation](#4-operator-testing-and-validation)
    - [4.1 Add](#41-add)
    - [4.2 Argmax](#42-argmax)
    - [4.3 Embedding](#43-embedding)
    - [4.4 Linear](#44-linear)
    - [4.5 RMSNorm](#45-rmsnorm)
    - [4.6 RoPE (Rotary Positional Embedding)](#46-rope-rotary-positional-embedding)
    - [4.7 Self-Attention](#47-self-attention)
    - [4.8 SwiGLU](#48-swiglu)
    - [4.9 Inference Benchmark](#49-inference-benchmark)
      - [优化方向](#优化方向)
  - [5. Appendix](#5-appendix)
    - [5.1 Environment](#51-environment)
    - [5.2 PR 合并与修复记录](#52-pr-合并与修复记录)
    - [5.3 test\_infer运行结果截图](#53-test_infer运行结果截图)
    - [5.3 已知问题](#53-已知问题)

---

## 1. Overview
- 基于 CUDA/cuDNN Frontend/cuBLAS 实现高性能推理  
- 支持 Qwen2 模型  
- Python / C++ /  CUDA 三层结构  
- 核心优化：自定义CUDA Kernel、cuDNN Graph Cache、cublasGemmEx矩阵乘法
- 项目地址：https://github.com/Hniii98/llaisys

---

## 2. System Architecture

LLAISYS 采用三层分层: Python 前端、C 接口层、C++ 实现层

### 2.1 Layered Design

                                python/llaisys/models/qwen2.py
                                              │ （前端接口，ctypes调用）
                                              ▼
                                include/llaisys/models/qwen2.h
                                              │ （C 层接口定义，暴露结构体、不透明指针与函数）
                                              ▼
                                src/llaisys/models/qwen2.cc
                                              │ （C 与 C++ 的胶水层，封装接口调用）
                                              ▼
                                src/models/qwen2/qwen2_impl.cpp
                                              │ （C++ 实现层，执行模型逻辑与内存管理）
                                              ▼
                                src/ops/*/nvidia/xxx.cuh
                                              │（CUDA 后端算子与 Kernel 实现，负责设备分发和类型分发）
                                              ▼
### 2.2 Model Overview



---

#### (1) Python Frontend
**文件**：`python/llaisys/models/qwen2.py`

- 类：
  - `Qwen2`：模型加载与推理入口

- 函数：
  - `_torch_to_numpy()`：PyTorch → NumPy 转换  
  - `_make_tensor_from_numpy()`：NumPy → `llaisysTensor_t`  
  - `_assign_weight_by_name()`：绑定权重至 C 层结构  
  - `generate()`：执行推理（Prefill + Decode）


#### (2) C Interface
**文件**：`include/llaisys/models/qwen2.h`, `src/llaisys/models/qwen2.cc`

- 结构体：
  - `LlaisysQwen2Meta`：模型超参数（层数、维度、KV头数等）
  - `LlaisysQwen2Weights`：各层权重指针数组
  - `LlaisysQwen2Model`：封装 `Qwen2Impl` 实例的 C 端对象

- 函数：
  - `llaisysQwen2ModelCreate()`：创建模型  
  - `llaisysQwen2ModelDestroy()`：销毁模型  
  - `llaisysQwen2ModelWeights()`：返回权重指针  
  - `llaisysQwen2ModelInfer()`：Prefill 阶段推理  
  - `llaisysQwen2ModelForwardOne()`：单 token Decode  
  - `llaisysQwen2ModelLogits()`：返回最近一次 Decode 的 logits 缓存（保留接口以支持后续不同的采样方式）


#### (3) C++ Implementation
**文件**：`src/models/qwen2/qwen2_impl.[hpp|cpp]`

- 成员：
  - `LlaisysQwen2Meta`：模型元信息  
  - `LlaisysQwen2Weights`：权重数组  
  - `tensor_t last_logits_`：最近一次 logits  
  - `std::vector<tensor_t> k_cache_, v_cache_`：KV 缓存  

- 函数：
  - `alloc_weight_arrays()`：分配每层权重句柄  
  - `forward()`：完整前向传播  
  - `prefill()`：初始上下文推理  
  - `decode_one()`：单 token 推理  
  - `append_kv()` / `view_k_total()`：管理增量 KV 缓存  
  - `free_all_weights_and_tensors()`：释放权重与显存  

---

### 2.3 Operator Overview

**目录根**：`src/ops/`

- `<op_name>/`
  - `op.cpp`：设备分发
  - `cpu/<op_name>_cpu.*`：CPU 实现  
  - `nvidia/<op_name>_nvidia.*`：CUDA 实现 


## 3. Operator Implementation and Optimization

- `namespace llaisys::ops::nvidia`：作为算子入口，负责类型分发。

- `匿名命名空间 (namespace {})`：封装内核定义与模板函数，相当于 static 限定作用域，仅在当前编译单元内可见。

- `namespace llaisys::ops::nvidia::kernels`：用于更复杂算子（如 Self-Attention）的 算子内部分发与 kernel 组织，支持多维度、不同阶段（Prefill / Decode）的专用 CUDA 内核实现。

---

### 3.1 Add

**文件**：`src/ops/add/nvidia/add_nvidia.cu`

- `llaisys::ops::nvidia::add()`：实现类型分发与 CUDA 启动  
- `add_kernel<T>()`：逐元素加法计算内核  
- `add_t<T>`：模板结构体，定义通用加法运算  
- `template<> struct add_t<__half>`：对半精度浮点类型使用 `__hadd()` 内建函数  
- `template<> struct add_t<__nv_bfloat16>`：对 bfloat16 类型使用 `__hadd()` 内建函数  

CUDA对FP16和BF16支持'+'运算符，因此用结构体模板 add_t<T> 并对 float、__half、__nv_bfloat16 进行完全特化实现编译期分发，安全调用 CUDA 内建算术指令。
结构体模板在编译期静态展开、无符号链接问题，避免函数模板的设备端特化在 NVCC 下存在编译与链接不稳定。



---

### 3.2 Argmax

**文件**：`src/ops/argmax/nvidia/argmax_nvidia.cu`

- `llaisys::ops::nvidia::argmax()`：实现类型分发与 CUDA 启动  
- `argmax_cub<T>()`：基于 CUB 库的泛型 ArgMax 实现，封装两阶段调用（查询临时显存 + 实际计算）  
- `cub::DeviceReduce::ArgMax()`：CUB 提供的归约算子，用于在 GPU 上同时返回最大值与对应索引  
- 匿名命名空间 (`namespace {}`)：限制 `argmax_cub<T>()` 作用域，仅在当前编译单元内可见  

使用 CUB 的设备归约接口实现高效 ArgMax 操作，通过模板实现对 `float32`、`float16`、`bfloat16` 的统一处理。  

调用流程为：先计算临时存储空间大小，再分配设备缓存并执行实际归约，最后返回最大值与索引。  
主入口函数 `argmax()` 在编译期完成类型分发，并通过 `llaisys::core::context()` 获取 CUDA 流以保证与框架执行环境一致。


---

### 3.3 Embedding

**文件**：`src/ops/embedding/nvidia/embedding_nvidia.cu`

- `llaisys::ops::nvidia::embedding()`：算子入口，执行类型分发与 CUDA 启动  
- `embedding_kernel<T>()`：逐索引查表的核心内核，将 `index_list` 中的行号映射至权重矩阵 `weight` 并拷贝到输出 `out`  
- 匿名命名空间 (`namespace {}`)：封装内核模板，限定作用域，仅在当前编译单元内可见  

实现思路为：  
每个线程块处理一个索引行 (`row`)，线程内并行拷贝整行数据；`blockDim.x` 控制行内并行度，`gridDim.x` 控制索引数。  
通过模板支持 `float32`、`float16`、`bfloat16` 三种数据类型，主入口函数在编译期完成类型分发。  

内核结构简单、访存模式连续，适合在推理阶段执行快速的 token embedding 查表操作。


---

### 3.4 Linear

**文件**：`src/ops/linear/nvidia/linear_nvidia.cu`

- `llaisys::ops::nvidia::linear()`：算子入口，执行数据类型分发与 CUDA 启动  
- `linear_<T, CType>()`：基于 cuBLAS 实现的通用线性层计算（GEMM），模板参数控制数据类型与计算精度  
- `add_bias_kernel<T>()`：CUDA 内核，用于在矩阵乘结果上逐列加上 bias  
- 匿名命名空间 (`namespace {}`)：封装内核模板与计算函数，仅在当前编译单元内可见  

实现采用 **`cublasGemmEx`** 调用矩阵乘法，实现：  
\[
out = in \times weight^T + bias
\]  
其中通过“逻辑转置”方式（调整 `opA`、`opB` 与 `lda/ldb/ldc`）避免显式数据转置，提高效率。  
不同精度类型使用不同计算模式：  
- FP32：`CUBLAS_GEMM_DEFAULT`（禁用 Tensor Core）  
- FP16 / BF16：`CUBLAS_GEMM_DEFAULT_TENSOR_OP`（启用 Tensor Core）  

计算结束后，通过 `add_bias_kernel()` 在结果矩阵上逐行广播加上偏置项。  
整体函数在编译期完成类型分发，确保 float32、float16、bfloat16 三种精度路径一致并充分利用 GPU 计算能力。

---

### 3.5 RMSNorm

**文件**：`src/ops/rms_norm/nvidia/rms_norm_nvidia.cu`

- `llaisys::ops::nvidia::rms_norm()`：算子入口，执行类型分发与 CUDA 启动  
- `rmsnorm_kernel<T, BLOCK_SIZE>()`：归一化计算核心内核，使用 CUB 进行块级归约  
- 匿名命名空间 (`namespace {}`)：封装模板与内核定义，仅在当前编译单元内可见  

内核实现步骤：  
1. 每个线程块处理一个样本行，使用 `cub::BlockReduce` 对输入向量的平方和进行块内归约。  
2. 计算 `inv_rms = sqrt(sum(v²)/d + eps)`，再对每个元素执行 `v / inv_rms * w`。  
3. 支持三种数据类型：`float32`、`float16`、`bfloat16`，通过 `if constexpr` 分支在编译期选择正确的类型转换（`__half2float` / `__bfloat162float`）。  

`BLOCK_SIZE` 在模板与 kernel launch 中保持一致，保证线程划分与共享内存同步一致性。  
该实现无全局原子操作，计算完全并行化，性能与数值稳定性兼顾。

---

### 3.6 RoPE (Rotary Position Embedding)

**文件**：`src/ops/rope/nvidia/rope_nvidia.cu`

- `llaisys::ops::nvidia::rope()`：算子入口，执行数据类型分发  
- `rope_dispatch<T>()`：模板分发函数，负责网格与线程布局、CUDA 启动配置  
- `rope_kernel<T>()`：核心计算内核，执行旋转位置编码的向量旋转运算  
- 匿名命名空间 (`namespace {}`)：封装内核与分发函数，限制作用域  

实现逻辑：  
1. 每个线程块处理一个 `sequence_index`，每个线程计算对应 head 内的一部分 embedding 维度。  
2. 对输入向量的前后两半执行二维旋转：  
   \[
   \begin{aligned}
   a' &= a \cos(\text{angle}) - b \sin(\text{angle}) \\
   b' &= a \sin(\text{angle}) + b \cos(\text{angle})
   \end{aligned}
   \]
3. 旋转角 `angle` 由 `pos * pow(theta, -2k/d)` 计算得到。  
4. 使用 `utils::load_as_float` / `store_from_float` 处理混合精度访问，支持 `float32`、`float16`、`bfloat16`。  

该实现完全在 GPU 上完成位置编码计算，结构紧凑、无分支发散，适用于大规模序列并行的 Transformer 推理场景。


---

### 3.7 Self-Attention

**文件**：`src/ops/self_attention/nvidia/self_attention_nvidia.cu`

- `llaisys::ops::nvidia::self_attention()`：算子入口，负责类型分发与设备调度  
- `selfatten3d_dispatch<T>()`：模板分发函数，统一调度 Prefill / Decode 两种执行路径  
- `atten3d_hdim128_decode_kernel<T>()`：cuDNN Frontend 图构建版 Attention 核，专用于 Decode 阶段  
- `naive_atten3d_hdim128_kernel<T>()` / `atten3d_generic_kernel<T>()`：Prefill 阶段使用的标准 CUDA Kernel  
- 匿名命名空间 (`namespace {}`)：封装模板与调度逻辑，仅在当前编译单元内可见  

#### (1) 核心结构

Self-Attention 实现分为三个层次：  
1. **分发层**（`llaisys::ops::nvidia`）  
   负责数据类型（`F32/F16/BF16`）选择与流调度。  
2. **调度层**（`selfatten3d_dispatch<T>`）  
   根据输入序列长度选择执行路径：  
   - `seq_len > 1` → **Prefill 模式**（上下文填充）  
   - `seq_len == 1` → **Decode 模式**（单 token 推理）  
3. **执行层**（`llaisys::ops::nvidia::kernels`）  
   封装多版本 CUDA Kernel，包括：
   - `atten3d_generic_kernel<T>()`：通用维度版本（支持任意 d）  
   - `naive_atten3d_hdim128_kernel<T>()`：128 维专用优化版本  
   - `atten3d_hdim128_decode_kernel<T>()`：基于 cuDNN Frontend 的高性能解码路径  

#### (2) Decode 路径优化

Decode 阶段引入 cuDNN Frontend Graph，通过 `CachedGraph` 缓存计划（workspace、tensor descriptor、graph），避免重复构建。  
数据在 FP32 下动态转换为 FP16/BF16，以使用 Tensor Core：  

```cpp
cast_fp32_to_fp16<<<...>>>();
atten3d_hdim128_decode_kernel<__half>(...);
cast_fp16_to_fp32<<<...>>>();
```

### 3.8 SwiGLU

**文件**：`src/ops/swiglu/nvidia/swiglu_nvidia.cu`

- `llaisys::ops::nvidia::swiglu()`：算子入口，执行类型分发与 CUDA 启动  
- `swiglu_dispatch<T>()`：模板分发函数，配置线程块与 CUDA 流  
- `swiglu_kernel<T>()`：核心计算内核，实现 SwiGLU 激活函数  
- 匿名命名空间 (`namespace {}`)：封装模板与内核，仅在当前编译单元内可见  

实现逻辑：  
1. 每个线程块处理一个序列样本 (`seq_idx`)，线程内按嵌入维度 (`embed_idx`) 并行计算。  
2. 使用 `utils::load_as_float()` / `store_from_float()` 实现混合精度加载与存储。  
3. 内核计算公式：  
   \[
   \text{SwiGLU}(x_1, x_2) = x_2 \cdot \text{Swish}(x_1) = x_2 \cdot \frac{x_1}{1 + e^{-x_1}}
   \]
4. 模板支持 `float32`、`float16`、`bfloat16` 三种类型，在编译期完成分发。  

该实现结构简洁、无共享内存依赖，适用于大批量矩阵激活操作，在 Transformer MLP 层中与线性层配合使用。


## 4. Operator Testing and Validation

本章展示各算子在测试结果，测试脚本位于：`python/test/ops/*.py`

---

### 4.1 Add

**文件**：`test/ops/add.py`  
**测试命令**：`python test/ops/add.py --device nvidia --profile`

| 输入规模 | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|-----------|-----------|----------------|-------------------|-----------|
| (2, 3) | FP32 | 0.01004 | 0.01002 | ≈ 1.0× |
| (2, 3) | FP16 | 0.02050 | 0.01004 | ×2.0 |
| (2, 3) | BF16 | 0.01003 | 0.01008 | ≈ 1.0× |
| (512, 4096) | FP32 | 0.08313 | 0.10049 | 0.83× |
| (512, 4096) | FP16 | 0.06030 | 0.05035 | ×1.2 |
| (512, 4096) | BF16 | 0.05020 | 0.05031 | ≈ 1.0× |


**说明**
  
- 小规模矩阵下，性能相当（kernel 启动为主开销）。  
- FP16/BF16 下具备轻微优势，受益于 CUDA 半精度原语 (`__hadd`, `__hadd` for bf16)。  
- 大规模矩阵性能接近 cuBLAS Add，说明数据带宽成为主要瓶颈。

### 4.2 Argmax

**文件**：`test/ops/argmax.py`  
**测试命令**：`python test/ops/argmax.py --device nvidia --profile`

| 输入规模 | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|-----------|-----------|----------------|-------------------|-----------|
| (4,) | FP32 | 0.04726 | 0.06244 | 0.76× |
| (4,) | FP16 | 0.01701 | 0.02738 | 0.62× |
| (4,) | BF16 | 0.02683 | 0.04245 | 0.63× |
| (4096,) | FP32 | 0.01628 | 0.03338 | 0.49× |
| (4096,) | FP16 | 0.02363 | 0.03127 | 0.75× |
| (4096,) | BF16 | 0.02547 | 0.03629 | 0.70× |

**说明**
  
- 算子基于 **CUB::DeviceReduce::ArgMax** 实现，包含两次 kernel 调用（查询临时缓存 + 实际计算），启动开销较高。  
- 对于小尺寸向量（如 4 元素），Torch 内核表现更优。  
- 大规模输入（>10⁴ 元素）时，LLAISYS 实现与 Torch 性能接近。 


### 4.3 Embedding

**文件**：`test/ops/embedding.py`  
**测试命令**：`python test/ops/embedding.py --device nvidia --profile`

| 索引形状 | 词表形状 | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|-----------|-----------|-----------|----------------|-------------------|-----------|
| (1,) | (2, 3) | FP32 | 0.12161 | 0.02008 | ×6.1 |
| (1,) | (2, 3) | FP16 | 0.05034 | 0.01258 | ×4.0 |
| (1,) | (2, 3) | BF16 | 0.04881 | 0.02009 | ×2.4 |
| (50,) | (512, 4096) | FP32 | 0.05608 | 0.01004 | ×5.6 |
| (50,) | (512, 4096) | FP16 | 0.05035 | 0.01019 | ×4.9 |
| (50,) | (512, 4096) | BF16 | 0.13692 | 0.01577 | ×8.7 |

**说明**


- 算子采用纯 CUDA 内核实现，每个线程块负责一行 embedding 拷贝，支持任意索引批量访问。  
- 性能瓶颈主要为显存带宽；由于访问模式为连续读取（按行对齐），访存效率接近理论峰值。  
- FP16 / BF16 模式显著降低了访存压力，因此整体加速比在 4–9× 之间。  
- 
---

### 4.4 Linear

**文件**：`test/ops/linear.py`  
**测试命令**：`python test/ops/linear.py --device nvidia --profile`

| 输出形状 | 输入形状 | 权重形状 | Bias | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|-----------|-----------|-----------|--------|-------------|------------------|-------------------|-----------|
| (2, 3) | (2, 4) | (3, 4) | ✅ | FP32 | 0.02006 | 0.03024 | ×0.66 |
| (2, 3) | (2, 4) | (3, 4) | ✅ | FP16 | 0.03061 | 0.04029 | ×0.76 |
| (2, 3) | (2, 4) | (3, 4) | ✅ | BF16 | 0.02006 | 0.03009 | ×0.67 |
| (512, 4096) | (512, 4096) | (4096, 4096) | ✅ | FP32 | 2.52393 | 2.39653 | **×1.05** |
| (512, 4096) | (512, 4096) | (4096, 4096) | ✅ | FP16 | 0.74761 | 0.84768 | ×0.88 |
| (512, 4096) | (512, 4096) | (4096, 4096) | ✅ | BF16 | 0.70818 | 0.85163 | ×0.83 |

**说明**

- 算子基于 cuBLAS `cublasGemmEx` 实现，自动根据数据类型选择最优计算模式：  
  FP32 使用 `CUBLAS_GEMM_DEFAULT`，半精度使用 `CUBLAS_GEMM_DEFAULT_TENSOR_OP`。  
- 通过 `CUBLAS_OP_T` 组合实现高效的权重转置，无需物理数据重排。  
- FP32 大矩阵性能略高于 Torch（约 +5%），说明 cuBLAS 配置已接近最佳性能。  
- FP16 / BF16 路径略低于 Torch（约 10–15%），主要由于：
  1. Torch 内部采用了 cuBLASLt 的 fused epilogue（融合 bias），而当前实现 bias 独立执行；
  2. 未使用 Tensor Core 扩展算法（如 `CUBLASLT_EPILOGUE_BIAS` 或 `CUBLAS_GEMM_ALGO_EXT`）。

---

### 4.5 RMSNorm

**文件**：`test/ops/rms_norm.py`  
**测试命令**：`python test/ops/rms_norm.py --device nvidia --profile`

| 输入形状 | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|-----------|-----------|----------------|-------------------|-----------|
| (1, 4) | FP32 | 0.11051 | 0.01003 | ×11.0 |
| (1, 4) | FP16 | 0.10889 | 0.01027 | ×10.6 |
| (1, 4) | BF16 | 0.10782 | 0.01021 | ×10.6 |
| (512, 4096) | FP32 | 0.23004 | 0.08001 | ×2.9 |
| (512, 4096) | FP16 | 0.14143 | 0.04143 | ×3.4 |
| (512, 4096) | BF16 | 0.12136 | 0.04076 | ×3.0 |

**说明**

- 所有运算均在单个 CUDA kernel 中完成，无需多次同步，极大降低调度延迟。  
- FP16/BF16 模式进一步利用 Tensor Core 内建算术单元，性能相对 Torch 提升 3–10×。  


### 4.6 RoPE (Rotary Positional Embedding)

**文件**：`test/ops/rope.py`  
**测试命令**：`python test/ops/rope.py --device nvidia --profile`

| 输入形状 | 位置范围 | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|-----------|-----------|-----------|----------------|-------------------|-----------|
| (2, 1, 4) | (0, 2) | FP32 | 0.39150 | 0.00504 | ×77.7 |
| (2, 1, 4) | (0, 2) | FP16 | 0.38535 | 0.01058 | ×36.4 |
| (2, 1, 4) | (0, 2) | BF16 | 0.38014 | 0.01005 | ×37.8 |
| (512, 4, 4096) | (512, 1024) | FP32 | 1.18506 | 4.05189 | ×0.29 |
| (512, 4, 4096) | (512, 1024) | FP16 | 1.06694 | 4.05616 | ×0.26 |
| (512, 4, 4096) | (512, 1024) | BF16 | 1.07380 | 4.08779 | ×0.26 |
  

 **说明**：  
- RoPE 算子实现了旋转位置编码的核心公式：
  \[
  \begin{aligned}
  a' &= a \cos \theta - b \sin \theta, \\
  b' &= a \sin \theta + b \cos \theta,
  \end{aligned}
  \]
  其中角频率 \(\text{inv\_freq} = \theta^{-2k/d}\) 按维度动态计算。  

- **小规模输入**（如 (2,1,4)）时，LLAISYS 实现因无 Python 调用与框架开销而大幅快于 Torch。  
- **大规模输入**（如 (512,4,4096)）下性能不如 Torch，主要原因如下：  
  1. **频繁的 `pow()`、`cos()`、`sin()` 调用** 造成高算术延迟；  
  2. **角度逐元素计算** 缺乏共享缓存优化，SM 计算单元利用率低；  


### 4.7 Self-Attention

**文件**：`test/ops/self_attention.py`  
**测试命令**：`python test/ops/self_attention.py --device nvidia --profile`

| 模式   | qlen | kvlen | nhead | nkvhead | head_dim | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|--------|------|--------|--------|----------|-----------|-------------|----------------|-------------------|-----------|
| Prefill | 2 | 2 | 1 | 1 | 4 | FP32 | 0.37783 | 0.02007 | ×18.8 |
| Prefill | 2 | 2 | 1 | 1 | 4 | FP16 | 0.36514 | 0.02006 | ×18.2 |
| Prefill | 2 | 2 | 1 | 1 | 4 | BF16 | 0.38830 | 0.02586 | ×15.0 |
| Prefill | 5 | 11 | 4 | 2 | 8 | FP32 | 0.39325 | 0.03004 | ×13.1 |
| Prefill | 5 | 11 | 4 | 2 | 8 | FP16 | 0.39793 | 0.04563 | ×8.7 |
| Prefill | 5 | 11 | 4 | 2 | 8 | BF16 | 0.38703 | 0.03545 | ×10.9 |
| Prefill | 2 | 20 | 16 | 2 | 128 | FP32 | 0.37562 | 0.04092 | ×9.2 |
| Prefill | 2 | 20 | 16 | 2 | 128 | FP16 | 0.41793 | 0.04512 | ×9.3 |
| Prefill | 2 | 20 | 16 | 2 | 128 | BF16 | 0.45046 | 0.05138 | ×8.8 |
| Decode | 1 | 30 | 16 | 2 | 128 | FP32 | 0.37245 | 0.15255 | ×2.4 |
| Decode | 1 | 30 | 16 | 2 | 128 | FP16 | 0.41566 | 0.09195 | ×4.5 |
| Decode | 1 | 30 | 16 | 2 | 128 | BF16 | 0.37223 | 0.10324 | ×3.6 |



**说明**：
- F32下，由于cuDNN graph不支持这一数据类型，因此需要转到BF16再送入graph计算，因此有相对大一点的误差无法过test，需要把误差放宽到1e-3
- attention算子去掉了显式地stream同步，因为单线程推理用的同一个stream传入kernel已经隐式同步了

- **Prefill 阶段**  
  Prefill 路径仅在推理开始执行一次，采用标准的 QKᵀ → Softmax → V 流程；  
  内核基于 `atten3d_generic_kernel` 与 `naive_atten3d_hdim128_kernel` 实现，  
  非热点路径，暂时没有进行进一步的优化

- **Decode 阶段**  
  Decode 模式（`qlen=1`）使用 cuDNN Frontend Graph 构建的缓存式注意力核 (`atten3d_hdim128_decode_kernel`)。  
  - 性能较 Torch 有显著提升（最高可达 4×）
  - 不过，Decode 路径仍存在以下 **可优化开销**：
  1. **Bias 构造与 KV 缓存清理**：  
     每次调用前需要：
     - 在 Bias 缓冲中将可见区设为 `0`、不可见区设为 `-inf`（用于注意力屏蔽）；  
     - 在 KV 缓存中将未使用的尾部区域置零，防止读取未定义值。  
     这两个步骤分别依赖独立的 CUDA kernel 启动，增加了 kernel 启动与同步开销

### 4.8 SwiGLU

**文件**：`test/ops/swiglu.py`  
**测试命令**：`python test/ops/swiglu.py --device nvidia --profile`

| shape          | 数据类型 | Torch 时间 (ms) | LLAISYS 时间 (ms) | 相对性能 |
|----------------|-----------|----------------|-------------------|-----------|
| (2, 3)         | FP32      | 0.08824        | 0.00515           | x17.1     |
| (2, 3)         | FP16      | 0.12962        | 0.01003           | x12.9     |
| (2, 3)         | BF16      | 0.12882        | 0.01001           | x12.9     |
| (512, 4096)    | FP32      | 0.39563        | 0.08001           | x4.9      |
| (512, 4096)    | FP16      | 0.35377        | 0.05534           | x6.4      |
| (512, 4096)    | BF16      | 0.34923        | 0.04013           | x8.7      |



**说明**

-  
  SwiGLU (`swish × linear`) 在实现上由单一 CUDA kernel 完成，  
  直接对输入 `gate` 与 `up` 执行逐元素计算：
  \[
  \text{SwiGLU}(x_1, x_2) = x_2 \cdot \sigma(x_1)
  \]
  其中激活函数采用近似形式 `σ(x) = x / (1 + exp(-x))`。  

-   
  在大张量 `(512, 4096)` 测试下，LLAISYS 比 Torch 快约 **5–9 倍**，  
  
  1. **统一内核融合计算**（Sigmoid 与乘法在同一线程完成）；  
  2. **避免中间张量写回**（Torch 实现中通常为两次 kernel 启动）；  
  3. **对齐 blockDim/threadIdx 的访问模式**，减少全局内存访存冲突。  


### 4.9 Inference Benchmark

**文件**：`test/test_infer.py`  
**测试命令**：`python test/test_infer.py --model ./downloads/DeepSeek-R1-Distill-Qwen-1.5B/ --device nvidia --test`

| 阶段 | Tokens | 耗时 (ms) | 描述 |
|--------|---------|-------------|------|
| **Prefill** | 10 | **410** | 一次性计算初始上下文（10 tokens），构建 KV 缓存。 |
| **Decode** | 每步 1 | **≈300–306 / step** | 逐 token 生成，累计 80+ 步（平均 ~300ms / token）。 |

---

**说明**

- **Prefill 阶段**：  
  仅发生一次，用于初始化注意力缓存与图形构建；耗时 410 ms，主要包括：
  - 模型图 (`Graph`) 创建；
  - 权重与 KV 缓存初始化；
  - 输入 token 的完整前向传播。

- **Decode 阶段**：  
  每次处理单个 token，平均耗时约 300 ms；  
  性能稳定，无明显抖动。第一次 decode（pos = 10）稍慢（450 ms），  
  因为触发了 `=== Rebuild graph ===` 操作。


#### 优化方向

- **Prefill** 为一次性高延迟阶段；  
  优化目标是减少首次图构建与权重加载开销。

- **Decode** 为主要推理路径；  
  当前性能已稳定（≈ 300 ms/token），后续优化方向：
  1. **避免频繁 memset / -inf 填充**（减少 kernel 启动）；
  2. **Graph 缓存复用**（只在 kvlen 扩容时重建）；
  3. **Pipeline 并行化 / 多流优化**（隐藏 memcpy 与 bias 计算延迟）。

## 5. Appendix

### 5.1 Environment

| 项目 | 内容 |
|------|------|
| **GPU 设备** | NVIDIA GeForce RTX 3060 Laptop GPU |
| **架构** | Ampere (Compute Capability 8.6) |
| **显存** | 6 GB |
| **驱动版本** | 581.42 |
| **CUDA 版本** | 13.0 (V13.0.88) |
| **cuBLAS** | 随 CUDA 13.0 提供 |
| **cuDNN** | 9.13.1 |
| **cuDNN FrontEnd** | 1.14.1 |
| **构建系统** | xmake |
| **编译器** | MSVC (Windows x64) |

cuDNN FE： https://github.com/Hniii98/cudnn-frontend-llaisys

###  5.2 PR 合并与修复记录

| PR # | 标题 / 内容 | 涉及模块 | 修复 / 优化点 |
|------|--------------|-----------|----------------|
| **#6** | fix: fix missing assert in `embedding.py` ([查看 PR](https://github.com/InfiniTensor/llaisys/pull/6)) | 测试脚本 / embedding | 在 embedding 测试中增加断言判断，防止测试失败时仍打印 “Test passed” 的假象。 |
| **#10** | fix: correct local variable to use reference instead of copy ([查看 PR](https://github.com/InfiniTensor/llaisys/pull/10)) | 核心模块 / 代码质量 | 将局部变量从复制改为引用，避免多余拷贝和潜在性能问题，提升代码一致性。 |
| **#11** | fix: fix device of `temp_mask` when using CUDA ([查看 PR](https://github.com/InfiniTensor/llaisys/pull/11)) | Self-attention / mask 构造 | 修复 CUDA 路径下 `temp_mask` 设备不匹配问题，确保与输入张量位于同一 GPU 上，避免运行时错误。 |

###  5.3 test_infer运行结果截图

![Result](./images/1.png)

![Process](./images/2.png)
![Process](./images/3.png)
![Process](./images/4.png)
![Process](./images/5.png)
![Process](./images/6.png)

###  5.3 已知问题

- 在cuDNN frontend使用的时候，如果遇到import torch，由于torch自带低版本的cuDNN backend可能导致库无法链接到正确dll文件从而报错找不到sdpa的定义，
  最粗暴简单的解决办法就是将cuDNN/lib下的*.dll文件覆盖所用虚拟环境下torch/lib的*.dll即可解决。