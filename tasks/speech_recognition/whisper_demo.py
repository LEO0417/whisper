#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper语音识别示例脚本
展示了更完整的Whisper模型使用方法，包括模型选择、语言设置等功能
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import get_device, download_model, create_pipeline, list_local_models

# Whisper模型选项
WHISPER_MODELS = {
    "tiny": {"name": "openai/whisper-tiny", "size": "~75MB", "quality": "最低", "speed": "最快"},
    "base": {"name": "openai/whisper-base", "size": "~140MB", "quality": "低", "speed": "快"},
    "small": {"name": "openai/whisper-small", "size": "~460MB", "quality": "中", "speed": "中"},
    "medium": {"name": "openai/whisper-medium", "size": "~1.5GB", "quality": "高", "speed": "慢"},
    "large": {"name": "openai/whisper-large-v3", "size": "~3GB", "quality": "最高", "speed": "最慢"}
}

def select_model():
    """交互式选择要使用的模型版本"""
    print("\n可选模型列表：")
    print("-" * 80)
    print(f"{'选项':<10}{'模型名称':<25}{'大小':<10}{'质量':<10}{'速度':<10}")
    print("-" * 80)
    
    for key, model in WHISPER_MODELS.items():
        print(f"{key:<10}{model['name']:<25}{model['size']:<10}{model['quality']:<10}{model['speed']:<10}")
    
    # 显示本地已有模型
    local_models = list_local_models()
    if local_models:
        print("\n本地已有模型：")
        for model in local_models:
            if "whisper" in model:
                print(f"- {model}")
    
    while True:
        choice = input("\n请选择要使用的模型 [默认 large]: ").strip().lower() or "large"
        if choice in WHISPER_MODELS:
            return WHISPER_MODELS[choice]["name"]
        print("无效选择，请重新输入")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Whisper语音识别示例")
    parser.add_argument("--audio", help="音频文件路径")
    parser.add_argument("--model", help="模型版本，可选：tiny, base, small, medium, large")
    parser.add_argument("--language", default="zh", help="音频语言，默认为中文(zh)")
    parser.add_argument("--download", action="store_true", help="下载模型到本地")
    parser.add_argument("--local_dir", help="本地模型保存目录")
    args = parser.parse_args()
    
    # 获取运行设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 选择模型
    if args.model and args.model in WHISPER_MODELS:
        model_name = WHISPER_MODELS[args.model]["name"]
    else:
        model_name = select_model()
    
    # 下载模型
    if args.download or input("\n是否下载模型到本地？(y/n) [默认n]: ").strip().lower() == "y":
        local_dir = args.local_dir or "./models/whisper"
        model_path = download_model(model_name, local_dir=local_dir)
    else:
        model_path = None
    
    # 创建语音识别pipeline
    pipe = create_pipeline(
        task="automatic-speech-recognition",
        model_name=model_name if not model_path else None,
        model_path=model_path
    )
    
    # 获取音频文件路径
    audio_file = args.audio
    if not audio_file:
        audio_file = input("请输入音频文件路径: ").strip()
    
    if not os.path.exists(audio_file):
        print(f"错误: 文件 '{audio_file}' 不存在")
        return
    
    print(f"正在处理音频文件: {audio_file}")
    print("处理中，请稍候...")
    
    # 执行语音识别
    result = pipe(audio_file, language=args.language)
    
    print("\n转录结果:")
    print("-" * 80)
    print(result["text"])
    print("-" * 80)
    
    # 询问是否将结果保存到文件
    if input("\n是否保存结果到文件？(y/n): ").strip().lower() == "y":
        output_file = os.path.splitext(audio_file)[0] + ".txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 