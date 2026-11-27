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

- sizePerThread=[1,8]：每个线程在列方向上按连续的 8 个元素进行分配。
- threadsPerWarp=[1,32]：一个 Warp 的 32 个 Lane 都沿列方向排列。
- warpsPerCTA=[1,4]：4 个 Warp 也沿列方向分布。
- order=[1,0]：列维度是优先连续布局的维度。

因此对于 BLOCK_M=4 的情况，同一组线程依次负责多行的数据，而不是每个 Warp 专门负责一行；这意味着多行计算可能提高每线程寄存器需求，而不一定能减少跨 warp reduction。

还有一个观察：两种版本都存在 ttg.convert_layout，用于将 Weight 的一维布局转换为适合二维运算的切片布局，但仅有 convert_layout 并不能证明它产生了昂贵的数据搬运。

这可以解释我们在 Experiment 006 中看到的现象：BLOCK_M=4 的 Registers/Thread 明显增加，并伴随 Occupancy 下降。
其原因是：Triton 没有改变 Warp 在行列维度上的分布，而是让同一组线程处理更多行的数据。这增加了每线程需要处理和可能同时保留的 x 数据，导致 Registers/Thread 从 63 增加到 164，进一步限制了 SM 可驻留的 CTA 数量。
