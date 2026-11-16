# Experiment 001 — Pytorch Eager vs Pytorch 官方 vs Triton

## 1. 目标

使用 `triton.testing.do_bench()` 比较用三种方式（Pytorch Eager vs Pytorch 官方 vs Triton）实现的 RMSNorm 的延迟。

## 2. 环境

- GPU: NVIDIA RTX 4090
- Shape: (1, 4096), (128, 4096), (1024, 4096), (128, 8192)
- dtype: FP16、BF16（CSV 中记为 `bfp16`）
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- Code commit: 1bf414aa94665852df54c4f291d42d1d21b7c60d

## 3. 实验结果

### FP16

| Shape | PyTorch Eager（reference）(μs) | PyTorch 官方 (μs) | Triton (μs) | 延迟比例（Eager : 官方 : Triton） |
|---|---:|---:|---:|---:|
| 1 × 4096 | 19.62 | 8.19 | 4.10 | 4.79 : 2.00 : 1 |
| 128 × 4096 | 32.61 | 9.22 | 6.14 | 5.31 : 1.50 : 1 |
| 1024 × 4096 | 93.18 | 25.66 | 20.48 | 4.55 : 1.25 : 1 |
| 128 × 8192 | 38.91 | 14.46 | 9.22 | 4.22 : 1.57 : 1 |

### BF16

| Shape | PyTorch Eager（reference）(μs) | PyTorch 官方 (μs) | Triton (μs) | 延迟比例（Eager : 官方 : Triton） |
|---|---:|---:|---:|---:|
| 1 × 4096 | 20.48 | 8.19 | 4.10 | 5.00 : 2.00 : 1 |
| 128 × 4096 | 32.77 | 9.22 | 7.04 | 4.65 : 1.31 : 1 |
| 1024 × 4096 | 90.94 | 26.62 | 20.48 | 4.44 : 1.30 : 1 |
| 128 × 8192 | 37.89 | 14.34 | 9.22 | 4.11 : 1.56 : 1 |

以上为各配置 do_bench 五轮测量的中位数，保留两位小数。数据来源：`results.csv`。延迟比例按未舍入的中位数计算，以 Triton 为 1。

注：CSV 中的 `bfp16` 表示 BF16（bfloat16）。

## 4. 观察

- 在给定 Shape 中， Triton 实现的延迟最低，其次是 Pytorch 官方（torch.nn.functional.rms_norm），再次是 Pytorch Eager。
- 随着 Shape 的增大，Pytorch 官方的实现和 Triton 实现的差距在减小。
- FP16 和 BF16 的实验结果相差不大。