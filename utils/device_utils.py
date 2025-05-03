import torch
import os

def get_device():
    """
    检测并返回最适合的设备
    
    Returns:
        str: 设备名称 ('mps', 'cuda', 'cpu')
    """
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    else:
        return "cpu"  # CPU

def print_device_info():
    """打印当前设备信息"""
    device = get_device()
    
    if device == "mps":
        print("使用 Apple MPS 设备进行加速")
    elif device == "cuda":
        print(f"使用 NVIDIA GPU 进行加速: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        print("使用 CPU 运行 (未检测到支持的 GPU)")
        
    print(f"PyTorch 版本: {torch.__version__}")
    
if __name__ == "__main__":
    # 测试设备检测
    print_device_info() 