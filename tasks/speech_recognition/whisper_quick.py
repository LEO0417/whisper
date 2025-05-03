#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper语音识别快速入门脚本
用于快速测试Whisper模型的语音识别功能
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import get_device, create_pipeline

def main():
    """主函数"""
    # 获取运行设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建语音识别pipeline
    pipe = create_pipeline(
        task="automatic-speech-recognition",
        model_name="openai/whisper-large-v3"
    )
    
    # 请求用户输入音频文件路径
    audio_file = input("请输入音频文件路径: ").strip()
    
    if not os.path.exists(audio_file):
        print(f"错误: 文件 '{audio_file}' 不存在")
        return
    
    print(f"正在处理音频文件: {audio_file}")
    print("处理中，请稍候...")
    
    # 执行语音识别
    result = pipe(audio_file)
    
    print("\n转录结果:")
    print("-" * 80)
    print(result["text"])
    print("-" * 80)
    
    print("\n提示:")
    print("- 如需更多功能，请尝试 whisper_demo.py 或 whisper_advanced.py")
    print("- 默认使用 'large-v3' 模型，提供最高精度但速度较慢")
    print("- 对于较短音频文件，可以考虑使用小型模型以提高速度")

if __name__ == "__main__":
    main() 