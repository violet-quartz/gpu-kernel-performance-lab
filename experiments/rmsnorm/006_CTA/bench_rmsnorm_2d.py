import argparse
import csv
from pathlib import Path

import torch
from kernels.triton.rmsnorm_2d import rmsnorm_triton_2d_into
import triton.testing

N = 4096
data_type = torch.float16
eps = 1e-6
return_mode = "median"

def main():
    parser = argparse.ArgumentParser(description="Profile the Triton RMSNorm kernel.")
    parser.add_argument(
        "--M",
        type=int,
        default=128
    )
    parser.add_argument(
        "--num_warps",
        type=int,
        default=4
    )
    parser.add_argument(
        "--block_m",
        type=int,
        default=2
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("results.csv")),
        help="CSV file for per-run latency results.",
    )
    args = parser.parse_args()
    M = args.M
    num_warps = args.num_warps
    block_m = args.block_m
    if M <= 0:
        parser.error("--M must be positive")
    if num_warps <= 0:
        parser.error("--num_warps must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn((M, N), generator=generator, dtype=data_type).to("cuda")
    weight = torch.randn((N,), generator=generator, dtype=data_type).to("cuda")
    output = torch.empty_like(x)

    # JIT warmup
    rmsnorm_triton_2d_into(
        x, weight, output,
        eps=eps,
        num_warps=num_warps,
        block_m=block_m,
        validate=False,
    )
    torch.cuda.synchronize()

    # CUDA Graph bench
    print(f'\nlatency median:')
    output_path = Path(args.output)
    with output_path.open("a", newline="", encoding="utf-8") as results_file:
        writer = csv.writer(results_file)
        if output_path.stat().st_size == 0:
            writer.writerow(("shape", "num_warps", "block_m", "run", "latency_us"))
        for run in range(1, 4):
            latency_us = triton.testing.do_bench_cudagraph(
                lambda: rmsnorm_triton_into(
                    x,
                    weight,
                    output,
                    eps=eps,
                    num_warps=num_warps,
                    block_m=block_m,
                    validate=False,
                ),
                rep=500,
                return_mode=return_mode,
            ) * 1000
            writer.writerow((f"{M}x{N}", num_warps, block_m, run, f"{latency_us:.6f}"))
            results_file.flush()
            print(f"{latency_us:.2f} us", end=' ')
    print('\n')

if __name__ == "__main__":
    main()
    