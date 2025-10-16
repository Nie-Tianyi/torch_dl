# CUDA Setup for PyTorch Project

This document describes how to configure CUDA support for this PyTorch project using UV package manager.

## Current Configuration

The project is configured to use CUDA 13.0 with the following setup:

- **PyTorch**: 2.9.0+cu130
- **TorchVision**: 0.24.0+cu130
- **CUDA Version**: 13.0
- **GPU**: NVIDIA GeForce RTX 3080

## Project Configuration

The project uses UV package manager with the following key configuration in `pyproject.toml`:

### Package Index Configuration
```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu130", marker = "sys_platform == 'windows'"},
    { index = "pytorch-cpu", marker = "sys_platform != 'windows'"},
]

torchvision = [
    { index = "pytorch-cu130", marker = "sys_platform == 'windows'"},
    { index = "pytorch-cpu", marker = "sys_platform != 'windows'"},
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

### Dependencies
```toml
dependencies = [
    "matplotlib>=3.10.5",
    "pandas>=2.3.2",
    "torch>=2.9.0",
    "torchvision>=0.24.0",
]
```

## Installation Steps

1. **Install CUDA dependencies manually** (if UV resolution fails):
   ```bash
   uv pip install torch==2.9.0+cu130 torchvision==0.24.0+cu130 --index-url https://download.pytorch.org/whl/cu130
   ```

2. **Update lock file**:
   ```bash
   uv lock
   ```

3. **Sync dependencies**:
   ```bash
   uv sync
   ```

## Verification

To verify CUDA is working correctly, run the test script:

```bash
python utils/test_cuda.py
```

This should output:
- PyTorch version with CUDA support
- CUDA availability status
- GPU device information
- Basic CUDA operations test

## Troubleshooting

### Common Issues

1. **UV resolution fails with platform compatibility errors**:
   - This is a known issue with UV when resolving PyTorch packages
   - Use the manual installation method above as a workaround

2. **CUDA not available**:
   - Check NVIDIA drivers are installed: `nvidia-smi`
   - Verify CUDA toolkit is installed
   - Ensure PyTorch CUDA version matches system CUDA version

3. **Performance issues**:
   - Ensure tensors are moved to GPU: `tensor.to('cuda')`
   - Use `torch.cuda.empty_cache()` to clear GPU memory if needed

## CUDA Usage in Code

```python
import torch

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Move tensors to GPU
x = torch.randn(3, 3).to(device)
```

## System Requirements

- **CUDA Version**: 13.0 or compatible
- **Python**: 3.13+
- **UV**: 0.8.15+
- **NVIDIA Drivers**: Latest compatible version

## Notes

- The project automatically uses CUDA on Windows and CPU on other platforms
- The configuration uses PyTorch's official CUDA 13.0 wheel repository
- Manual installation may be required due to UV's platform resolution limitations
