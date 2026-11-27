# Experiment 007 - Triton Layout & Generated Code Analysis

## 1. 目标

在 x 的 shape 为 (4096, 4096) 的情况下，比较二维 RMSNorm 在 BLOCK_M=1 和 BLOCK_M=4 时的 TTGIR、PTX 与寄存器使用情况，理解 Triton 如何将逻辑 Tensor 映射到 Threads 和 Warps。

## 2. 环境

- GPU: NVIDIA RTX 6000 Ada Generation
- Shape: [4096, 4096]
- dtype: FP16
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- ncu version: 2025.3.1.0 (build 36398880) (public-release)
- Code commit: 1cc42a5119c8c2845f59598225129000a8df2482

## 3. 实验

```shell
PYTHONPATH=. .venv/bin/python \
    experiments/rmsnorm/007_ttgir/compile_rmsnorm_2d.py --BLOCK_M=1
PYTHONPATH=. .venv/bin/python \
    experiments/rmsnorm/007_ttgir/compile_rmsnorm_2d.py --BLOCK_M=4

grep -nE '#blocked|sizePerThread|threadsPerWarp|warpsPerCTA|convert_layout' \
    rmsnorm_m1.ttgir rmsnorm_m4.ttgir
```

实验结果：

| 指标 | BLOCK_M=1 | BLOCK_M=4 |
| --- | --- | --- |
| Registers/Thread | 63 | 164 |
| Shared Memory | 16 B | 64 B |
| Tensor Shape | [1, 4096] | [4, 4096] |
| sizePerThread | [1, 8] | [1, 8] |
| threadsPerWarp | [1, 32] | [1, 32] |
| warpsPerCTA | [1, 4] | [1, 4] |
| order | [1, 0] | [1, 0] |


