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
    # warmup
    rmsnorm_triton_into(x, weight, output, eps)
    torch.cuda.synchronize()
    rmsnorm_triton_into(x, weight, output, eps, validate=False)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()