from kernels.triton.rmsnorm_2d import rmsnorm_2d_kernel
import argparse
import torch
import triton

M, N = 4096, 4096
dtype = torch.float16
eps = 1e-6
num_warps = 4

def main():
    parser = argparse.ArgumentParser(description="Profile the Triton RMSNorm kernel.")
    parser.add_argument(
        "--BLOCK_M",
        type=int,
        default=1,
        choices=[1, 4]
    )
    args = parser.parse_args()
    BLOCK_M = args.BLOCK_M

    BLOCK_N = triton.next_power_of_2(N)
    grid = (triton.cdiv(M, BLOCK_M),)
    compiled = rmsnorm_2d_kernel.warmup(
        dtype, dtype, dtype, # x、output、weight 的 dtype
        eps, M, N,
        grid=grid,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )

    with open(f"rmsnorm_m{BLOCK_M}.ttgir", "w") as f:
        f.write(compiled.asm["ttgir"])

    with open(f"rmsnorm_m{BLOCK_M}.ptx", "w") as f:
        f.write(compiled.asm["ptx"])

    # Load the binary so the driver can report register usage without a launch.
    compiled._init_handles()
    print(f"register count={compiled.n_regs}")
    print(f"shared memory={compiled.metadata.shared} bytes")
    print(f"available asm: {list(compiled.asm.keys())}")

if __name__ == '__main__':
    main()