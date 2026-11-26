import argparse

import torch
from kernels.triton.rmsnorm_2d import rmsnorm_triton_2d_into

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
    parser.add_argument(
        "--num_warps",
        type=int,
        default=4
    )
    parser.add_argument(
        "--BLOCK_M",
        type=int,
        default=1
    )
    args = parser.parse_args()
    shape = tuple(args.shape)
    if any(dim <= 0 for dim in shape):
        parser.error("--shape dimensions must be positive")
    num_warps = args.num_warps
    if num_warps <= 0:
        parser.error("--num_warps must be positive")
    block_m = args.BLOCK_M

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=data_type).to("cuda")
    weight = torch.randn((shape[-1],), generator=generator, dtype=data_type).to("cuda")
    output = torch.empty_like(x)
    # warmup
    rmsnorm_triton_2d_into(x, weight, output, eps=eps, num_warps=num_warps, block_m=block_m)
    torch.cuda.synchronize()
    rmsnorm_triton_2d_into(x, weight, output, eps=eps, num_warps=num_warps, block_m=block_m, validate=False)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()