import torch
from kernels.triton.rmsnorm import rmsnorm_triton_into

shape = (128, 4096)
data_type = torch.float16
eps = 1e-6

def main():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=data_type).to("cuda")
    weight = torch.randn((shape[-1],), generator=generator, dtype=data_type).to("cuda")
    output = torch.empty_like(x)

    # JIT warmup
    rmsnorm_triton_into(x, weight, output, eps)
    torch.cuda.synchronize()

    # CUDA graph capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            rmsnorm_triton_into(
                x, weight, output, eps,
                validate=False
            )
    torch.cuda.synchronize()

    # Normal Launch
    torch.cuda.nvtx.range_push("normal_launch")
    for _ in range(100):
        rmsnorm_triton_into(
            x, weight, output, eps,
            validate=False
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # CUDA Graph Replay
    torch.cuda.nvtx.range_push("cuda_graph_replay")
    g.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()