import torch


def test_cuda():
    """Test CUDA availability and basic functionality"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")

        # Test basic CUDA operations
        device = torch.device("cuda")
        print(f"Using device: {device}")

        # Create a tensor on GPU
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)

        # Perform operations on GPU
        z = torch.matmul(x, y)

        print(f"Tensor x shape: {x.shape}")
        print(f"Tensor y shape: {y.shape}")
        print(f"Result z shape: {z.shape}")
        print("CUDA test completed successfully!")

        # Test memory usage
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    else:
        print("CUDA is not available. Please check your installation.")


if __name__ == "__main__":
    test_cuda()
