# check_torch.py
import torch

print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available (PyTorch can see GPU): {is_cuda_available}")

if is_cuda_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(0)}")
    # 这一行显示的是PyTorch在编译时使用的CUDA版本
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
else:
    print("\n[诊断]: PyTorch未能检测到任何可用的CUDA设备。")
    print("[原因]: 这通常意味着您安装的是CPU版本的PyTorch，或者NVIDIA驱动与PyTorch所需的CUDA版本不兼容。")