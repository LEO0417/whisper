from huggingface_hub import snapshot_download
import os
from transformers import pipeline
from .device_utils import get_device

def download_model(model_name, local_dir=None):
    """
    下载模型到本地
    
    Args:
        model_name (str): 模型名称，如 "openai/whisper-large-v3"
        local_dir (str, optional): 本地保存目录。如果为None，则使用默认缓存目录
        
    Returns:
        str: 模型保存路径
    """
    print(f"正在下载模型: {model_name}")
    
    if local_dir:
        # 确保目录存在
        os.makedirs(local_dir, exist_ok=True)
        # 下载模型到指定目录
        model_path = snapshot_download(repo_id=model_name, local_dir=local_dir)
    else:
        # 使用默认缓存目录
        model_path = snapshot_download(repo_id=model_name)
        
    print(f"模型已下载到: {model_path}")
    return model_path

def create_pipeline(task, model_name=None, model_path=None):
    """
    创建指定任务的pipeline
    
    Args:
        task (str): 任务类型，如 "automatic-speech-recognition", "question-answering" 等
        model_name (str, optional): 模型名称，如果指定则使用该模型
        model_path (str, optional): 本地模型路径，如果指定则优先使用本地模型
        
    Returns:
        pipeline: 创建的pipeline实例
    """
    device = get_device()
    
    if model_path:
        # 使用本地模型文件
        print(f"使用本地模型: {model_path}")
        pipe = pipeline(task, model=model_path, device=device)
    elif model_name:
        # 使用指定模型
        print(f"使用模型: {model_name}")
        pipe = pipeline(task, model=model_name, device=device)
    else:
        # 使用任务默认模型
        print(f"使用默认模型")
        pipe = pipeline(task, device=device)
        
    return pipe

def list_local_models(base_dir=None):
    """
    列出本地已下载的模型
    
    Args:
        base_dir (str, optional): 要检查的基础目录，如果为None则检查默认缓存目录
        
    Returns:
        list: 本地模型列表
    """
    local_models = []
    
    # 检查默认缓存目录
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # 如果指定了基础目录，则检查该目录
    if base_dir and os.path.exists(base_dir):
        for model_dir in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, model_dir)):
                local_models.append(model_dir)
    
    # 检查缓存目录
    if os.path.exists(cache_dir):
        for model_dir in os.listdir(cache_dir):
            if model_dir.startswith("models--") and os.path.isdir(os.path.join(cache_dir, model_dir)):
                # 从目录名称提取模型名称
                parts = model_dir.split("--")
                if len(parts) >= 3:
                    model_name = f"{parts[1]}/{parts[2]}"
                    if model_name not in local_models:
                        local_models.append(model_name)
    
    return local_models 