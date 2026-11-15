import argparse
import csv

import triton.testing
import torch
from references.rmsnorm import rmsnorm_reference
from kernels.triton.rmsnorm import rmsnorm_triton, rmsnorm_triton_into

# shapes = [(1, 4096), (16, 4096), (128, 4096), (1024, 4096), (128, 8192)]
shapes = [(1, 4096), (1024, 4096)]
# data_types = [torch.float16, torch.bfloat16]
data_types = [torch.float16]
fns = [rmsnorm_reference, rmsnorm_triton]
eps = 1e-6
return_mode = 'median'


def main():
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm implementations.")
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="CSV file for per-run latency results (default: benchmark_results.csv)",
    )
    args = parser.parse_args()

    fields = ("shape", "dtype", "implementation", "method", "run", "latency_us")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)

    with open(args.output, "w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fields)
        writer.writeheader()
        for shape in shapes:
            for dtype in data_types:
                x = torch.randn(shape, generator=generator, dtype=dtype).to("cuda")
                weight = torch.randn(shape[-1], generator=generator, dtype=dtype).to("cuda")
                output = torch.empty_like(x)
                dtype_name = (
                    str(dtype)
                    .removeprefix("torch.")
                    .replace("float16", "fp16")
                    .replace("bfloat16", "bf16")
                )
                print(f"\n\nShape: {shape}, dtype: {dtype}", end=" ")
                implementations = [
                    (fn.__name__.removeprefix("rmsnorm_"), lambda fn=fn: fn(x, weight, eps))
                    for fn in fns
                ]
                implementations.append(
                    (
                        "torch.nn.functional.rms_norm",
                        lambda: torch.nn.functional.rms_norm(
                            x, (shape[-1],), weight=weight, eps=eps
                        ),
                    )
                )
                implementations.append(
                    (
                        "triton_into",
                        lambda: rmsnorm_triton_into(
                            x, weight, output, eps
                        ),
                    )
                )
                for implementation, call in implementations:
                    print(f"\n{implementation}:")
                    for method, benchmark in (
                        (
                            "do_bench",
                            lambda: triton.testing.do_bench(
                                call,
                                warmup=100,
                                rep=500,
                                return_mode=return_mode,
                            ),
                        ),
                        (
                            "cudagraph",
                            lambda: triton.testing.do_bench_cudagraph(
                                call,
                                rep=500,
                                return_mode=return_mode,
                            ),
                        ),
                    ):
                        print(f"{method} median:", end=" ")
                        for run in range(1, 6):
                            latency_us = benchmark() * 1000
                            writer.writerow(
                                {
                                    "shape": f"{shape[0]}x{shape[1]}",
                                    "dtype": dtype_name,
                                    "implementation": implementation,
                                    "method": method,
                                    "run": run,
                                    "latency_us": f"{latency_us:.6f}",
                                }
                            )
                            results_file.flush()
                            print(f"{latency_us:.2f} us", end=" ")
                print("\n")


if __name__ == "__main__":
    main()
