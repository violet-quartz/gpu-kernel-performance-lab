import argparse

import torch
from kernels.triton.rmsnorm import rmsnorm_triton_into

data_type = torch.float16
eps = 1e-6

def main():
    parser = argparse.ArgumentParser(description="Profile the Triton RMSNorm kernel.")
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=(128, 4096),
        metavar=("ROWS", "COLS"),
        help="Input shape (default: 128 4096).",
    )
    args = parser.parse_args()
    shape = tuple(args.shape)
    if any(dim <= 0 for dim in shape):
        parser.error("--shape dimensions must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=data_type).to("cuda")
    weight = torch.randn((shape[-1],), generator=generator, dtype=data_type).to("cuda")
    output = torch.empty_like(x)
    # warmup
    rmsnorm_triton_into(x, weight, output, eps)
    torch.cuda.synchronize()
    rmsnorm_triton_into(x, weight, output, eps, validate=False)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()