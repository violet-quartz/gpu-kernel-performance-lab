# Environment

## 运行环境

```
python -c "import torch, triton; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('Triton:', triton.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用')"
```

PyTorch: 2.13.0+cu130
PyTorch CUDA: 13.0
Triton: 3.7.1
CUDA available: True
GPU: NVIDIA GeForce RTX 4090

## 安装测试依赖

在已安装好与机器 CUDA 环境匹配的 PyTorch 和 Triton 的 Python 环境中，安装测试依赖并运行测试：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

`requirements-dev.txt` 用于声明测试工具；PyTorch 和 Triton 的安装方式可能因 CUDA 环境而异，需单独选择合适的版本。