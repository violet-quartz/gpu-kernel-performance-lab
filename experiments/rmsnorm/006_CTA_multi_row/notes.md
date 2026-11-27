# Experiment 006 - Optimization: CTA Multi-Row

## 1. 目标

验证在一个 CTA 中处理多行，能否在大 shape 下提升性能表现，猜想是处理多行能够复用 weight，减少内存的读取，当然 weight 的读取很可能从缓存中读取，所以本猜想不一定成立。

## 2. 环境

- GPU: NVIDIA RTX 6000 Ada Generation
- Shape: [128, 4096], [4096, 4096], [16384, 4096]
- dtype: FP16
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- ncu version: 2025.3.1.0 (build 36398880) (public-release)
- Code commit: 1cc42a5119c8c2845f59598225129000a8df2482

## 3. 实验

### benchmark 指标

```shell
for M in 128 4096 16384; do
    for BLOCK_M in 1 2 4; do
        echo "shape=($M, 4096), BLOCK_M=$BLOCK_M"
        PYTHONPATH=. python \
        experiments/rmsnorm/006_CTA_multi_row/bench_rmsnorm_2d.py \
            --M "$M" \
            --BLOCK_M "$BLOCK_M"
    done
done
```

| Shape | BLOCK_M=1 | BLOCK_M=2 | BLOCK_M=4 |
|---|---:|---:|---:|
| 128 × 4096 | 1.67 μs | 1.60 μs | 1.97 μs |
| 4096 × 4096 | 15.92 μs | 16.29 μs | 16.66 μs |
| 16384 × 4096 | 330.35 μs | 333.76 μs | 330.49 μs |

以上延迟为每种配置三次运行结果的中位数。

看起来 Multi-Row 并没有带来性能收益，我们以 4096 × 4096 为例用 ncu 分析一下原因。

### NCU 指标（Shape=4096 × 4096）

```shell
for BLOCK_M in 1 2 4; do
    PYTHONPATH=. ncu \
        --set basic \
        --metrics lts__t_sectors_aperture_device_op_read.sum,sm__sass_inst_executed_op_local_ld.sum,sm__sass_inst_executed_op_local_st.sum \
        --launch-count 1 \
        -o /tmp/rmsnorm_multi_row_full_metrics_${BLOCK_M} \
        python profiles/profile_rmsnorm_2d.py \
        --shape 4096 4096 --BLOCK_M "$BLOCK_M"
done
```

| 配置 | Registers/Thread | L2 Read Sectors | SM Throughput | L2 Throughput | Duration (μs) | Register Spilling | Achieved Occupancy |
|---|---:|---:|---:|---:|---:|---|---:|
| Baseline 单行 (BLOCK_M=1) | 63 | 1,090,064 | 19.44% | 39.13% | 44.00 | 否（0 local LD/ST） | 58.98% |
| BLOCK_M=2 | 96 | 1,088,818 | 15.47% | 37.85% | 45.82 | 否（0 local LD/ST） | 37.11% |
| BLOCK_M=4 | 164 | 1,091,246 | 14.01% | 35.92% | 48.10 | 否（0 local LD/ST） | 22.92% |

注：L2 Read Sectors 使用 `lts__t_sectors_aperture_device_op_read.sum`（单位：sector）；寄存器溢出依据 `sm__sass_inst_executed_op_local_ld.sum` 和 `sm__sass_inst_executed_op_local_st.sum` 判断，两者均为 0。SM 和 L2 吞吐率分别对应报告中的 `Compute (SM) Throughput` 和 `L2 Cache Throughput`。

主要观察：

在 [4096,4096] 下，随着 BLOCK_M 从 1 增加到 4，NCU Kernel Duration 从 44.00 μs 增加至 48.10 μs，性能下降约 9.3%。

Registers/Thread 从 63 增加至 164，Achieved Occupancy 从 58.98% 降低至 22.92%。对于 BLOCK_M=4，寄存器资源将每个 SM 的理论 Resident CTA 数量限制为 3。

三种配置的 L2 Read Sectors 基本相同，说明 Multi-row 的 Weight 复用没有带来明显的 L2 Read Traffic 减少。

L2 Throughput 从 39.13% 降至 35.92%，但这并不代表 L2 Traffic 减少。结合 Duration 增加，更合理的解释是单位时间内处理的 L2 请求量下降。

未观察到 Register Spilling，因此性能下降不能归因于寄存器溢出产生的 Local Memory 访问。

结论：

对于当前 RMSNorm 实现，Multi-row 没有带来稳定的性能收益。

实验结果支持以下解释：增加每个 CTA 处理的行数提高了 Register Pressure，限制 SM 同时驻留的 CTA 和 Warp 数量，而 Weight 复用没有明显降低 L2 Read Traffic，因此整体性能没有得到改善。

本实验说明，减少逻辑上的 Weight Load 或 CTA 数量，不一定能够提升 Kernel 性能；优化效果需要结合实际 Memory Traffic、资源占用和 Kernel Duration 综合判断。