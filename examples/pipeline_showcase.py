#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformers Pipeline 展示脚本

这个脚本展示了多种 Hugging Face Transformers pipeline 的使用方法，
包括语音识别、问答、文本生成、翻译和会话等任务。
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import get_device, create_pipeline, print_device_info

def showcase_asr():
    """展示语音识别 (Automatic Speech Recognition)"""
    print("\n=== 语音识别示例 ===")
    
    # 创建 pipeline
    pipe = create_pipeline(
        task="automatic-speech-recognition",
        model_name="openai/whisper-tiny"  # 使用小模型以加快演示速度
    )
    
    # 请求用户输入
    audio_file = input("请输入音频文件路径：").strip()
    
    if not os.path.exists(audio_file):
        print(f"错误：文件 '{audio_file}' 不存在")
        return
    
    # 执行识别
    print("处理中...")
    result = pipe(audio_file)
    
    # 显示结果
    print("\n识别结果：")
    print("-" * 50)
    print(result["text"])
    print("-" * 50)

def showcase_qa():
    """展示问答系统"""
    print("\n=== 问答系统示例 ===")
    
    # 创建 pipeline
    pipe = create_pipeline(
        task="question-answering",
        model_name="distilbert-base-cased-distilled-squad"
    )
    
    # 示例上下文和问题
    context = """
    人工智能（英语：Artificial Intelligence, AI）是由人类研发的机器所表现出的智能。
    从字面上理解，人工智能就是人造的智能，但实际上没有一个被普遍接受的定义。
    通常认为，人工智能是指由人制造出来的机器所表现出来的智能，这类机器叫做智能体。
    目前比较普遍的观点认为，把人工智能定义为研究智能体的计算机科学分支较为合适。
    """
    
    question = input("请输入问题 (例如：什么是人工智能？): ").strip() or "什么是人工智能？"
    
    # 执行问答
    print("思考中...")
    result = pipe(question=question, context=context)
    
    # 显示结果
    print("\n问答结果：")
    print("-" * 50)
    print(f"问题：{question}")
    print(f"回答：{result['answer']}")
    print(f"置信度：{result['score']:.4f}")
    print("-" * 50)

def showcase_text_generation():
    """展示文本生成"""
    print("\n=== 文本生成示例 ===")
    
    # 创建 pipeline
    pipe = create_pipeline(
        task="text-generation",
        model_name="gpt2"
    )
    
    # 示例提示
    prompt = input("请输入提示文本 (例如：人工智能将在未来): ").strip() or "人工智能将在未来"
    
    # 执行生成
    print("生成中...")
    result = pipe(
        prompt,
        max_length=50,
        temperature=0.7,
        num_return_sequences=1,
        do_sample=True
    )
    
    # 显示结果
    print("\n生成结果：")
    print("-" * 50)
    print(result[0]['generated_text'])
    print("-" * 50)

def showcase_translation():
    """展示翻译"""
    print("\n=== 翻译示例 ===")
    
    # 创建 pipeline
    pipe = create_pipeline(
        task="translation",
        model_name="Helsinki-NLP/opus-mt-zh-en"  # 中译英
    )
    
    # 示例文本
    text = input("请输入中文文本 (例如：人工智能是计算机科学的一个分支): ").strip() or "人工智能是计算机科学的一个分支"
    
    # 执行翻译
    print("翻译中...")
    result = pipe(text)
    
    # 显示结果
    print("\n翻译结果：")
    print("-" * 50)
    print(f"原文：{text}")
    print(f"译文：{result[0]['translation_text']}")
    print("-" * 50)

def showcase_conversation():
    """展示会话"""
    print("\n=== 会话示例 ===")
    
    # 创建 pipeline
    pipe = create_pipeline(
        task="conversational",
        model_name="facebook/blenderbot-400M-distill"
    )
    
    # 示例对话
    print("\n请进行对话，最多三轮 (输入 'q' 提前结束):")
    
    conversation = None
    
    for i in range(3):
        user_input = input("\n用户：").strip()
        if user_input.lower() == 'q':
            break
            
        # 执行对话
        if conversation is None:
            conversation = pipe(user_input)
        else:
            conversation = pipe(user_input, past_user_inputs=conversation.past_user_inputs, 
                               generated_responses=conversation.generated_responses)
            
        # 显示结果
        bot_response = conversation.generated_responses[-1]
        print(f"机器人：{bot_response}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Transformers Pipeline 展示")
    parser.add_argument("--task", choices=["asr", "qa", "text", "translation", "conversation", "all"],
                       help="要展示的任务 (默认：all)")
    args = parser.parse_args()
    
    # 打印设备信息
    print("\n=== Transformers Pipeline 展示 ===")
    print_device_info()
    
    # 确定要展示的任务
    tasks = []
    if args.task and args.task != "all":
        tasks = [args.task]
    else:
        tasks = ["asr", "qa", "text", "translation", "conversation"]
    
    # 执行展示
    for task in tasks:
        if task == "asr":
            showcase_asr()
        elif task == "qa":
            showcase_qa()
        elif task == "text":
            showcase_text_generation()
        elif task == "translation":
            showcase_translation()
        elif task == "conversation":
            showcase_conversation()
    
    print("\n=== 展示完成 ===")
    print("如需了解更多详情，请查看 'tasks/' 目录下的各个任务脚本")

if __name__ == "__main__":
    main() 