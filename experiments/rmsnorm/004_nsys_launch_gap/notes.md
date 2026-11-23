# Experiment 004 — nsys Launch Gap Analysis

## 1. 目标

在实验二中，针对  cuda graph benchmark 耗时比 normal launch 耗时低的现象，我们提出了假设 H1: CPU 提交速率可能影响普通 Benchmark 的测量结果。这里我们通过 nsys 对两种运行方式进行耗时分析，看假设是否成立，并获得其他相关信息。

## 2. 环境

- GPU: NVIDIA RTX 4090
- Shape: [128, 4096]
- dtype: FP16
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- Code commit: 3f4d3ff31c3e88535389b633ab727f082d6db17a

## 3. 实验过程

```bash
# 安装 nsight-systems
apt-get update
apt-get install -y nsight-systems-cli

nsys --version
nsys status -e

# 采集 timeline
# --trace=cuda,nvtx：采集 CUDA API、GPU Kernel 和 NVTX
# --cuda-graph-trace=node： 让 Nsys 采集 CUDA Graph 内部的 Kernel 节点，便于逐个计算 Kernel Duration 和 Gap。
# --sample=none：禁用 CPU Sampling，--cpuctxsw=none：禁用 CPU Context Switch Trace，减少权限问题，对于 CPU 具体的开销我们先不采集。
PYTHONPATH=. nsys profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    -o /tmp/rmsnorm_launch_gap \
    python experiments/rmsnorm/004_nsys_launch_gap/nsys_profile_launch_gap.py

# 查看汇总
nsys stats \
    --report cuda_gpu_kern_sum,cuda_api_sum \
    /tmp/rmsnorm_launch_gap.nsys-rep

# 导出 sqlite
nsys export \
    --type sqlite \
    --output /tmp/rmsnorm_launch_gap.sqlite \
    --force-overwrite=true \
    /tmp/rmsnorm_launch_gap.nsys-rep

# 使用脚本进一步分析 sqlite 数据
PYTHONPATH=. \
    python experiments/rmsnorm/004_nsys_launch_gap/data_analysis.py
```

## 4. 实验结果

| 类别 | 指标 | 平均值（µs） |
|---|---|---:|
| Normal Launch Kernel | Duration | 1.956800 |
| Normal Launch Kernel | Gap | 39.082667 |
| CUDA Graph Kernel | Duration | 1.782720 |
| CUDA Graph Kernel | Gap | 0.045899 |
| cuLaunchKernelEx API | Duration | 8.685080 |
| cuLaunchKernelEx API | Launch Interval | 41.154010 |
| cuLaunchKernelEx API | Between API | 32.461909 |

## 5. 结论

Normal Launch Kernel 和 CUDA Graph Kernel，Gap（start - 上一个 kernel 的 end）差距明显，为 39.08 µs vs 0.046 µs，Duration 差距较小，为 1.96 µs vs 1.78 µs。

CPU 两次 cuLaunchKernelEx 开始时刻的间隔为 41.15 µs，约等于 Normal Lauch Kernel Gap + Duration（39.08 µs + 1.96 µs），能够支持 H1：当前短 Kernel 的 GPU 执行节奏主要受 CPU 逐次提交节奏限制。

另外，两次 cuLaunchKernelEx 开始时刻的间隔中， API 的调用 duration 只有 8.69 µs，说明 API 调用本身不是 CPU 侧唯一的时间成本，还有相当多的 CPU 时间。

Normal Launch：约 95.2% 的序列时间花在 Kernel 之间。CUDA Graph：约 97.5% 的序列时间用于 Kernel 执行，Gap 约占 2.5%。

另外，为了排除 Nsys 追踪插桩开销的影响，我们用 python 的 time.perf_counter() 分别统计 Normal Launch Kernel 和 CUDA Graph Kernel 运行 1000 次的耗时（experiments/rmsnorm/004_nsys_launch_gap/python_profile_launch_gap.py），结果为 Normal Launch 16.749 ms，CUDA Graph 1.746 ms，CUDA Graph 批量执行加速 9.60x，graph replay 的耗时确实显著降低。 受 Nsys 插桩的影响，上面 nsys 的结果并不等同于 Python Wrapper 的真实固有开销。






