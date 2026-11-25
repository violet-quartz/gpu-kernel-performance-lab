# Experiment 003 - ncu baseline

## 1. 目标

## 2. 环境

- GPU: NVIDIA RTX 6000 Ada Generation
- Shape: [128, 4096]
- dtype: FP16
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- Code commit: f58b504ee2108d7a9be7f61aedd933a1fd430996

## 3. 实验结果

```shell
PYTHONPATH=. ncu \
    --set basic \
    --launch-count 1 \
    -o rmsnorm_ncu_test \
    python experiments/rmsnorm/003_ncu_baseline/profile_rmsnorm.py

ncu --import rmsnorm_ncu_test.ncu-rep \
    --page details \
    --print-details all
```

实验结果：
| 指标 | 实测值 |
|---|---:|
| GPU | RTX 6000 Ada |
| Kernel Duration | 5.54 μs |
| SM Throughput | 5.25% |
| Memory / DRAM Throughput | 21.42% |
| L1/TEX Throughput | 18.32% |
| L2 Throughput | 13.65% |
| Theoretical Occupancy | 66.67% |
| Achieved Occupancy | 8.36% |
| Grid Size | 128 Blocks |
| Block Size | 128 Threads |
| Registers / Thread | 62 |
| SM 数量 | 142 |
| Waves Per SM | 0.11 |

在 shape 为 [128, 4096] 的情况下，Achieved Occupancy 为 8.38%，远小于 Theoretical Occupancy，且Waves Per SM （整个 Grid 的 CTA 数量相对于 GPU 一次能容纳 CTA 的数量比）为 0.11 较低，都说明当前的并行度太低。下面我们保持 Triton 算子代码不动，改变 M 的值看一下测量结果。

```shell
for M in 1 128 256 1024 4096; do
    PYTHONPATH=. ncu \
        --set basic \
        --launch-count 1 \
        --force-overwrite \
        -o "rmsnorm_m${M}" \
        .venv/bin/python \
        experiments/rmsnorm/003_ncu_baseline/profile_rmsnorm.py \
        --shape "$M" 4096
done
```
实验结果：
| Shape | Duration | Achieved Occupancy | SM Throughput | DRAM Throughput | L2 Throughput |
|---|---:|---:|---:|---:|---:|
| 1 × 4096 | 4.32 μs | 8.39% | 0.05% | 0.59% | 0.63% |
| 128 × 4096 | 5.54 μs | 8.09% | 5.28% | 21.32% | 13.65% |
| 256 × 4096 | 7.33 μs | 13.70% | 7.76% | 31.96% | 17.45% |
| 1024 × 4096 | 16.38 μs | 48.81% | 13.26% | 56.55% | 26.90% |
| 4096 × 4096 | 44.29 μs | 60.61% | 19.53% | 83.28% | 39.30% |


## 4. 观察

- M=1、128 时，GPU 总吞吐利用率很低；M 增大后，Effective Bandwidth 和资源利用率明显提高。
- M=4096 时，DRAM Throughput 已达到 83.28%，明显高于其他 Shape。

## 5. 提出假设

H1：小 Shape 受到并行工作量不足的限制

H2：大 Shape 可能逐渐受到 DRAM 带宽限制

H1 从上面的实验中已经得到了支持。下面我们关注一下 Memory Throughput，验证一下 H2。

## 6. 验证

```shell
for M in 128 4096; do
    PYTHONPATH=. ncu \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section LaunchStats \
        --section Occupancy \
        --cache-control all \
        --launch-count 1 \
        --force-overwrite \
        -o "/tmp/rmsnorm_memory_m${M}" \
        .venv/bin/python \
        experiments/rmsnorm/003_ncu_baseline/profile_rmsnorm.py \
        --shape "$M" 4096
done

for M in 128 4096; do
    PYTHONPATH=. ncu \
        --metrics \
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct \
        --cache-control all \
        --launch-count 1 \
        --force-overwrite \
        -o "/tmp/rmsnorm_traffic_m${M}" \
        .venv/bin/python \
        experiments/rmsnorm/003_ncu_baseline/profile_rmsnorm.py \
        --shape "$M" 4096
done
```

实验结果：
| 指标 | M=128 | M=4096 |
|---|---:|---:|
| DRAM Read Bytes | 1.06 MB | 33.57 MB |
| DRAM Write Bytes | 0 B | 0 B |
| L2 Hit Rate | 68.05% | 51.08% |
| L2 Read/Write Traffic | Read 68,265 sectors (2,184,480 B) / Write 32,867 sectors (1,051,744 B) | Read 1,090,777 sectors (34,904,864 B) / Write 1,049,164 sectors (33,573,248 B) |
| DRAM Throughput | 21.32% | 83.28% |
| Kernel Duration | 5.54 μs | 44.29 μs |

从上面的结果中，我们看到当前 Kernel 的 DRAM Write Bytes 为 0， 因此我们认为输出写入主要先保留在 L2。

下面再做两个实验：

实验一：
将 --cache-control all 变成 --cache-control none。all 会在每次 Kernel Replay 前刷新 GPU 缓存，提高跨 Replay 的可重复性；none 不执行这种刷新，指标更依赖先前 GPU 工作及 Replay 顺序。

```shell
PYTHONPATH=. ncu \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct \
    --cache-control none \
    --launch-count 1 \
    --force-overwrite \
    -o /tmp/rmsnorm_memory_m4096_no_cache_control_full \
    .venv/bin/python experiments/rmsnorm/003_ncu_baseline/profile_rmsnorm.py \
    --shape 4096 4096
```

| 指标（M=4096） | cache-control all | cache-control none | 变化 |
|---|---:|---:|---:|
| DRAM Read Bytes | 33.57 MB | 25.38 MB | -8.19 MB |
| DRAM Write Bytes | 0 B | 4.16 MB | +4.16 MB |
| L2 Hit Rate | 51.08% | 88.83% | +37.75 个百分点 |
| L2 Read/Write Traffic | Read 1,090,777 sectors (34,904,864 B) / Write 1,049,164 sectors (33,573,248 B) | Read 1,087,807 sectors (34,809,824 B) / Write 1,049,351 sectors (33,579,232 B) | Read -2,970 sectors (-95,040 B) / Write +187 sectors (+5,984 B) |
| DRAM Throughput | 83.28% | 78.42% | -4.86 个百分点 |
| Kernel Duration | 44.29 μs | 41.41 μs | -2.88 μs（约快 6.5%） |

可以看到 Kernel Replay 前不刷新缓存的话，L2 Hit Rate 增高，与预期相同，受缓存策略的影响，DRAM Read/Write Bytes 和 L2 Read/Write Traffic 都有所变化，cache-control 的设置会对结果产生影响。

实验二：
将 M 进一步增大到 8192 和 16384，统一使用 `--cache-control all`：

```shell
for M in 4096 8192 16384; do
    PYTHONPATH=. ncu \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct \
        --cache-control all \
        --launch-count 1 \
        --force-overwrite \
        -o "/tmp/rmsnorm_memory_m${M}_cache_control_all_full" \
        .venv/bin/python \
        experiments/rmsnorm/003_ncu_baseline/profile_rmsnorm.py \
        --shape "$M" 4096
done
```

| 指标（cache-control all） | M=4096 | M=8192（相对 M=4096） | M=16384（相对 M=4096） |
|---|---:|---:|---:|
| DRAM Read Bytes | 33.57 MB | 67.13 MB（+33.56 MB） | 134.24 MB（+100.67 MB） |
| DRAM Write Bytes | 0 B | 14.57 MB（+14.57 MB） | 71.58 MB（+71.58 MB） |
| L2 Hit Rate | 51.10% | 50.60%（-0.50 个百分点） | 50.38%（-0.72 个百分点） |
| L2 Read/Write Traffic | Read 1,090,462 sectors (34,894,784 B) / Write 1,049,371 sectors (33,579,872 B) | Read 2,143,934 sectors (68,605,888 B；+1,053,472 sectors / +33,711,104 B) / Write 2,097,992 sectors (67,135,744 B；+1,048,621 sectors / +33,555,872 B) | Read 4,249,738 sectors (135,991,616 B；+3,159,276 sectors / +101,096,832 B) / Write 4,194,714 sectors (134,230,848 B；+3,145,343 sectors / +100,650,976 B) |
| DRAM Throughput | 82.50% | 92.65%（+10.15 个百分点） | 90.53%（+8.03 个百分点） |
| Kernel Duration | 44.70 μs | 96.83 μs（+52.13 μs，+116.6%） | 249.60 μs（+204.90 μs，+458.4%） |

可以看出，随着 M 的增大，DRAM Read/Write Bytes 不断增大，DRAM Throughput 接近峰值并略有下降（Kernel Duration 以比 DRAM Read/Write Bytes 更高的比例增大），说明当 M = 16384 时，已经受到了 DRAM 带宽的限制，“H2：大 Shape 可能逐渐受到 DRAM 带宽限制”得到了验证。


