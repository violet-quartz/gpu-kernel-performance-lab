# Experiment 002 — CUDA Graph Benchmark

## 1. 目标

比较 `triton.testing.do_bench()` 和 `do_bench_cudagraph()` 测得的 RMSNorm 延迟，分析两种测量方法的差异。

## 2. 环境

- GPU: NVIDIA RTX 4090
- Shape: [1, 4096], [1024, 4096]
- dtype: FP16
- Software versions: PyTorch: 2.13.0+cu130, PyTorch CUDA: 13.0, Triton: 3.7.1
- Code commit: 1bf414aa94665852df54c4f291d42d1d21b7c60d

## 3. 实验结果

| Shape | Implementation | do_bench (μs) | CUDA Graph (μs) |
|---|---|---:|---:|
| 1 × 4096 | Triton | 4.10 | 1.14 |
| 1 × 4096 | PyTorch 官方 | 8.19 | 3.03 |
| 1024 × 4096 | Triton | 20.48 | 5.92 |
| 1024 × 4096 | PyTorch 官方 | 26.62 | 6.77 |

以上为各配置五轮测量的中位数。

## 4. 观察

- CUDA Graph Benchmark 测得的延迟显著低于普通 do_bench。
    - 对于 [1,4096]，Triton 的 CUDA Graph 延迟约为普通 Benchmark 的 27.8%，Pytorch 官方的 CUDA Graph 延迟约为普通 Benchmark 的 37%
    - 对于 [1024, 4096]，Triton 的 CUDA Graph 延迟约为普通 Benchmark 的 28.9%，Pytorch 官方的 CUDA Graph 延迟约为普通 Benchmark 的 25.4%
- 当 Shape 增大时，Triton 与 PyTorch 官方实现的相对性能差距缩小。

## 5. 提出假设

为什么 CUDA Graph Benchmark 测得的延迟显著低于普通 do_bench 呢？
- H1：CPU 提交（CPU Launch Overhead）速率可能影响普通 Benchmark 的测量结果。
- H2：两种 Benchmark 的 Cache Flush 和数据复用行为可能不同。
- H3：Output Allocation 和 Python validation 可能影响实验结果。

## 6. 证据与限制

### H1: CPU 提交速率可能影响普通 Benchmark 的测量结果

CUDA Graph 的重要作用是减少 CPU 每次提交 GPU 工作的成本。使用 CUDA Graph 减少了 CPU Launch Overhead，使得延迟降低。

```
CPU → Launch Kernel 1
CPU → Launch Kernel 2
CPU → Launch Kernel 3
```

```
CPU → Replay Graph
             ├── Kernel 1
             ├── Kernel 2
             └── Kernel 3
```

### H2: 两种 Benchmark 的 Cache Flush 和数据复用行为可能不同

```python
import inspect
import triton.testing

print(inspect.getsource(triton.testing.do_bench))
print(inspect.getsource(triton.testing.do_bench_cudagraph))
```

通过上面的代码，打印两种 Benchmark 的具体实现：

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

可以看出在重复实验时，do_bench 每次运行函数前都会清理 L2 cache，而 do_bench_cudagraph 则没有这个操作，因此可以推断 do_bench_cudagraph 在连续执行相同 Kernel 时，数据更有机会留在 L2 中，而 do_bench 则会更频繁的从显存读取数据，导致 do_bench 测得的延迟增加。

### H3: Output Allocation 和 Python validation 可能影响实验结果

查看 CUDA Graph 的捕获方式：捕获时 Python 函数会执行一次； `g.replay()` 时只重放捕获的 GPU 工作，不会重新执行整个 Python 函数。

对于 `rmsnorm_triton`，

| 部分 | 捕获阶段 | 每次 g.replay() |
|---|---|---|
| torch.empty_like(x) | 创建 Tensor，分配输出显存 | 复用同一块显存，不重新创建 Tensor |
| Python validation，例如检查 shape、dtype、device | 执行检查 | 不再执行 |
| Triton kernel | 记录 GPU 操作及参数 | 再次执行 kernel |
| return output | 返回 Python Tensor 对象 | 不执行 Python return |

我们可以把创建输出内存和 python validation 进行预先处理，然后在循环中只启动同一个 RMSNorm Kernel，
因此我们对代码进行重构，区分了 `rmsnorm_triton` 和 `rmsnorm_triton_into` 函数，然后在 validate=false 的情况下测量一下 rmsnorm_triton_into 的耗时。

```
def rmsnorm_triton_into(
    x:torch.Tensor, # [M, N]
    weight: torch.Tensor, # [N]
    output: torch.Tensor, # [M, N]
    eps: float = 1e-6,
    validate: bool = True):
    if validate:
        validate_rmsnorm_inputs(x, weight, eps)
        if not x.is_cuda:
            raise ValueError("x must be on CUDA")
        validate_rmsnorm_output(output, x, weight)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_kernel[grid](x, output, weight, eps, N, BLOCK_SIZE)

def rmsnorm_triton(
    x: torch.Tensor, # [M, N]
    weight: torch.Tensor, # [N]
    eps: float = 1e-6
) -> torch.Tensor:
    output = torch.empty_like(x)
    rmsnorm_triton_into(x, weight, output, eps)
    return output
```



## 7. Next Actions



## 8. Conclusion

CUDA Graph Benchmark 和普通 do_bench 测得的延迟存在显著差异，说明测量方式会影响短 Kernel 的性能结果。

目前尚未确定差异的具体来源，需要进一步 Profiling。