# RMSNorm

```
rmsnorm(input, weight, eps=1e-6)
```

Apply Root Mean Square Layer Normalization over the last dimension of a two dimension Tensor.

Root Mean Squre Layer Normalization:

$$
y_i = \frac{x_i}{\operatorname{RMS}(x)}\gamma_i
$$
$$ where $$
$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}x_i^2+\epsilon}
$$

- $N$：归一化维度大小，也就是 Hidden Size。
- $x_i$：输入元素。
- $\gamma_i$：可学习的缩放参数（weight）。
- $\epsilon$：数值稳定项。

## Parameters
- input (torch.Tensor): 2D Tensor with shape [M, N].
- weight (torch.Tensor): 1D Tensor with shape [N].
- eps (float, optional): a value added to the denominator for numerical stability. Default: 1e-6.

## Output
- output (torch.Tensor): the same shape and dtype with input.

## Other
| 属性              | 第一版约束                 |
| --------------- | --------------------- |
| Tensor Layout   | Contiguous            |
| Device          | CUDA                  |
| dtype           | FP32、FP16、BF16. The same dtype for input and weight.       |
| Accumulation    | FP32                  |
|                 |                       |
