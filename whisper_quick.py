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