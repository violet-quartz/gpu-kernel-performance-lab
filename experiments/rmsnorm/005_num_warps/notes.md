# Experiment 005 - Optimization: num_warps

## 1. 目标

调整 num_warps，看 rmsnorm 的性能有何变化，并通过 ncu profile 分析原因。

## 2. 环境

- GPU: NVIDIA RTX 6000 Ada Generation
- Shape: [128, 4096], [16384, 4096]
- dtype: FP16
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- ncu version: 2025.3.1.0 (build 36398880) (public-release)
- Code commit: f58b504ee2108d7a9be7f61aedd933a1fd430996

## 3. 实验

```shell
for M in 128 16384; do
    for W in 1 2 4 8; do
        PYTHONPATH=. python \
        experiments/rmsnorm/005_num_warps/bench_rmsnorm.py \
        --M "$M" \
        --num_warps "$W"
    done
done
```

运行结果：

| Shape | 1 Warp | 2 Warps | 4 Warps | 8 Warps |
|---|---:|---:|---:|---:|
| 128 × 4096 | 2.045 μs | 1.823 μs | 1.629 μs | 1.621 μs |
| 16384 × 4096 | 332.523 μs | 328.169 μs | 330.236 μs | 326.634 μs |

M=128 时，1 warp 相对 4 warps 耗时增加 25.5%，差异明显；
M=16384 时， 8 warps 相对 4 warps 耗时降低 −1.09%，略有下降；
下面我们就用 ncu 来分析一下这些差异。

```shell
for config in "128 1" "128 4" "16384 4" "16384 8"; do
    read M W <<< "$config"

    PYTHONPATH=. ncu \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --cache-control all \
    --launch-count 1 \
    --force-overwrite \
    -o "/tmp/rmsnorm_m${M}_w${W}" \
    .venv/bin/python \
    profiles/profile_rmsnorm.py \
    --shape "$M" 4096 \
    --num_warps "$W"
done
```

### NCU 指标

| 指标 | M=128 W=1 | M=128 W=4 | M=16384 W=4 | M=16384 W=8 |
|---|---:|---:|---:|---:|
| Kernel Duration (μs) | 6.82 | 5.47 | 250.14 | 249.95 |
| Registers/Thread | 195 | 62 | 62 | 40 |
| Registers/CTA（计算） | 6,240 | 7,936 | 7,936 | 10,240 |
| Theoretical Occupancy | 16.67% | 66.67% | 66.67% | 100% |
| Achieved Occupancy | 2.08% | 8.11% | 63.27% | 92.98% |
| Waves Per SM | 0.11 | 0.11 | 14.42 | 19.23 |
| SM Throughput | 3.29% | 5.24% | 14.15% | 16.26% |
| DRAM Throughput | 17.48% | 21.42% | 90.34% | 90.62% |
| DRAM Read Bytes | 1.07 MB | 1.06 MB | 134.24 MB | 134.24 MB |
| DRAM Write Bytes | 0 B | 0 B | 71.62 MB | 72.10 MB |

Registers/CTA 按 `Registers/Thread × CTA threads` 计算。

从上表可以看出：
M=128 时，W=1 显著慢于 W=4，每线程工作量与寄存器需求明显增加（但无寄存器溢出证据）；小 Shape 需要关注单 CTA 执行效率；
M=16384 时，W=8 显著提高 Occupancy，但 DRAM Throughput 和 Duration 几乎不变；更多 warp 无法显著突破当前的内存带宽限制。

下面我们看一下 M=128 时 W=1 为什么慢，查看 InstructionStats 和 WarpStateStats 这两个 Section。

```shell
for W in 1 4; do
    PYTHONPATH=. ncu \
        --section InstructionStats \
        --section WarpStateStats \
        --cache-control all \
        --launch-count 1 \
        --force-overwrite \
        -o "/tmp/rmsnorm_instruction_w${W}" \
        .venv/bin/python \
        profiles/profile_rmsnorm.py \
        --shape 128 4096 \
        --num_warps "$W"
done
```

- num_warps=1 的 Registers/Thread 为 195，明显高于 num_warps=4 的 62，但两者均未观察到 Register Spilling。

- Kernel 总执行指令数分别为 116,736 和 140,800。按启动的 Warp 数归一化，平均每 Warp 执行指令数分别约为 912 和 275，W=1 约为 W=4 的 3.32 倍。


这些结果支持以下解释：W=4 将每行计算分配给更多线程和 Warp，降低每 Warp 的指令工作量，提供更多并行执行机会，因此在当前小 Shape 下表现出更低的执行延迟。

