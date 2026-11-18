import time
import torch
from kernels.triton.rmsnorm import rmsnorm_triton_into
import statistics

shape = (128, 4096)
data_type = torch.float16
eps = 1e-6

def measure_total_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()

    fn()

    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000
 

def main():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=data_type).to("cuda")
    weight = torch.randn((shape[-1],), generator=generator, dtype=data_type).to("cuda")
    output = torch.empty_like(x)

    def normal_launch():
        for _ in range(1000):
            rmsnorm_triton_into(
                x, weight, output, eps, validate=False
            )

    # JIT warmup
    rmsnorm_triton_into(x, weight, output, eps)
    torch.cuda.synchronize()

    # CUDA graph capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        normal_launch()
    torch.cuda.synchronize()

    # Graph Replay Warmup
    g.replay()
    torch.cuda.synchronize()

    def graph_replay():
        g.replay()

    normal_launch_time = []
    graph_replay_time = []
    for _ in range(5):
        t1 = measure_total_time(normal_launch)
        t2 = measure_total_time(graph_replay)
        normal_launch_time.append(t1)
        graph_replay_time.append(t2)
    
    print("Normal launch (ms):", normal_launch_time)
    print("Graph replay (ms):", graph_replay_time)

    normal_median = statistics.median(normal_launch_time)
    graph_median = statistics.median(graph_replay_time)

    print(f"Normal median: {normal_median:.3f} ms")
    print(f"Graph median: {graph_median:.3f} ms")

    print(f"Normal per kernel: {normal_median:.3f} us")
    print(f"Graph per kernel: {graph_median:.3f} us")

    print(f"Speedup: {normal_median / graph_median:.2f}x")

if __name__ == '__main__':
    main()
