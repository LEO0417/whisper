import test_torch
from transformers import pipeline
from huggingface_hub import snapshot_download
import os

# 检查 MPS 是否可用（适用于 Apple Silicon 芯片）
device = "mps" if test_torch.backends.mps.is_available() else "cpu"
print(f"使用设备：{device}")

# 下载并缓存模型到本地
def download_model():
    """下载模型到本地 Hugging Face 缓存目录"""
    
    # 下载模型文件
    model_path = snapshot_download(repo_id="openai/whisper-large-v3", local_dir="./whisper-large-v3")
    print(f"模型已下载到：{model_path}")
    return model_path

# 创建语音识别 pipeline
def create_pipeline(model_path=None):
    """创建语音识别 pipeline"""
    if model_path:
        # 使用本地模型文件
        pipe = pipeline("automatic-speech-recognition", 
                       model=model_path,
                       device=device)
    else:
        # 使用在线模型
        pipe = pipeline("automatic-speech-recognition", 
                       model="openai/whisper-large-v3",
                       device=device)
    return pipe

# 处理音频文件
def transcribe_audio(pipe, audio_file):
    """转录音频文件"""
    print(f"正在处理音频文件：{audio_file}")
    result = pipe(audio_file)
    return result

def select_model():
    """交互式选择要下载的模型版本"""
    models = {
        "1": {"name": "openai/whisper-tiny", "size": "~75MB", "quality": "最低", "speed": "最快"},
        "2": {"name": "openai/whisper-base", "size": "~140MB", "quality": "低", "speed": "快"},
        "3": {"name": "openai/whisper-small", "size": "~460MB", "quality": "中", "speed": "中"},
        "4": {"name": "openai/whisper-medium", "size": "~1.5GB", "quality": "高", "speed": "慢"},
        "5": {"name": "openai/whisper-large-v3", "size": "~3GB", "quality": "最高", "speed": "最慢"}
    }
    
    print("\n可选模型列表：")
    print("-" * 80)
    print(f"{'编号':<6}{'模型名称':<25}{'大小':<10}{'质量':<10}{'速度':<10}")
    print("-" * 80)
    for key, model in models.items():
        print(f"{key:<6}{model['name']:<25}{model['size']:<10}{model['quality']:<10}{model['speed']:<10}")
    
    # 检查本地已有模型
    local_models = []
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        for model_key in models.values():
            model_name = model_key["name"].split("/")[1]
            potential_paths = [
                os.path.join(cache_dir, f"models--openai--{model_name}"),
                f"./whisper-models/{model_name}"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    local_models.append(model_key["name"])
                    break
    
    if local_models:
        print("\n本地已有模型：")
        for model in local_models:
            print(f"- {model}")
    
    while True:
        choice = input("\n请选择要下载的模型编号 (1-5) [默认 5]: ").strip() or "5"
        if choice in models:
            return models[choice]["name"]
        print("无效选择，请重新输入")

# 示例用法
if __name__ == "__main__":
    # 下载模型到本地
    local_model_path = download_model()
    
    # 创建 pipeline
    pipe = create_pipeline(local_model_path)
    
    # 处理示例音频（这里需要有一个音频文件）
    # 您可以替换为自己的音频文件路径
    # 支持的格式：wav, mp3, ogg 等
    audio_file = "example_audio.mp3"  # 替换为你的音频文件
    
    try:
        result = transcribe_audio(pipe, audio_file)
        print("\n转录结果：")
        print(result["text"])
    except FileNotFoundError:
        print(f"文件未找到：{audio_file}")
        print("请将示例代码中的音频文件路径替换为真实存在的音频文件")
        
    print("\n使用方法指南：")
    print("1. 将音频文件路径修改为您要转录的文件")
    print("2. 模型已下载到本地，可随时离线使用")
    print("3. 对于较长的音频，处理可能需要一些时间，请耐心等待") 