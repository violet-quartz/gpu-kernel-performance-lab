# Experiment 002 — CUDA Graph Benchmark

## 1. Objective

比较 `triton.testing.do_bench()` 和 `do_bench_cudagraph()` 测得的 RMSNorm 延迟，分析两种测量方法的差异。

## 2. Environment

- GPU: NVIDIA RTX 4090
- Shape: [1, 4096], [1024, 4096]
- dtype: FP16
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- Code commit: 1bf414aa94665852df54c4f291d42d1d21b7c60d

## 3. Results

| Shape | Implementation | do_bench (μs) | CUDA Graph (μs) |
|---|---|---:|---:|
| 1 × 4096 | Triton | 4.10 | 1.14 |
| 1 × 4096 | PyTorch 官方 | 8.19 | 3.03 |
| 1024 × 4096 | Triton | 20.48 | 5.92 |
| 1024 × 4096 | PyTorch 官方 | 26.62 | 6.77 |

以上为各配置五轮测量的中位数。

## 4. Observations（实验观察）

- CUDA Graph Benchmark 测得的延迟显著低于普通 do_bench。
    - 对于 [1,4096]，Triton 的 CUDA Graph 延迟约为普通 Benchmark 的 27.8%。
    - 对于 [1024, 4096]，
- 当 Shape 增大时，Triton 与 PyTorch 官方实现的相对性能差距缩小。

## 5. Hypotheses（待验证假设）

- H1：CPU 提交速率可能影响普通 Benchmark 的测量结果。
- H2：两种 Benchmark 的 Cache Flush 和数据复用行为可能不同。
- H3：Output Allocation 和地址复用方式可能影响实验结果。

## 6. Evidence and Limitations（证据与限制）

```python
import inspect
import triton.testing

print(inspect.getsource(triton.testing.do_bench))
print(inspect.getsource(triton.testing.do_bench_cudagraph))
```

```python
def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)

def do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    import torch
    assert return_mode in ["min", "max", "mean", "median", "all"]

    with torch.cuda.stream(torch.cuda.Stream()):
        # warmup
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        # step 1 - we estimate the amount of time the kernel call takes
        # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
        #       but it is probably good enough
        # NOTE: we don't use a graph to estimate the runtime because creating a graph is expensive,
        #       ~300ms on A100, so we default to the same method used in `do_bench` (minus the L2
        #       cache flush).
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        # Rewrite to avoid possible division by 0 issues with fast benchmarks
        if estimate_ms == 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, int(rep / estimate_ms))
        # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
        # host overhead
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                fn()
        torch.cuda.synchronize()
        # measure time and return
        ret = []
        n_retries = 10
        for _ in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            ret += [start_event.elapsed_time(end_event) / n_repeat]
        return _summarize_statistics(ret, quantiles, return_mode)
```

目前只有延迟测量，没有 Nsight Systems 或 Nsight Compute 数据。

因此尚不能确定三个假设各自贡献了多少性能差异，也不能将两种 Benchmark 的时间差直接解释为 CPU Launch Overhead。

## 7. Next Actions



## 8. Conclusion

CUDA Graph Benchmark 和普通 do_bench 测得的延迟存在显著差异，说明测量方式会影响短 Kernel 的性能结果。

目前尚未确定差异的具体来源，需要进一步 Profiling。