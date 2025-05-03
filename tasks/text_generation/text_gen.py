#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本生成示例脚本
展示如何使用Transformers的text-generation pipeline创建文本生成应用
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import get_device, create_pipeline

# 默认文本生成模型
DEFAULT_MODEL = "gpt2"

def interactive_generation(pipe):
    """交互式文本生成"""
    print("\n欢迎使用文本生成系统！")
    print("输入'退出'或'exit'结束使用\n")
    
    while True:
        # 获取用户输入
        prompt = input("请输入提示文本: ").strip()
        if prompt.lower() in ["退出", "exit"]:
            break
        
        # 获取生成参数
        try:
            max_length = int(input("最大生成长度 [默认 50]: ").strip() or "50")
            temperature = float(input("温度参数(0.1-1.0) [默认 0.7]: ").strip() or "0.7")
            num_return = int(input("生成数量 [默认 1]: ").strip() or "1")
        except ValueError:
            print("输入参数格式错误，使用默认值")
            max_length = 50
            temperature = 0.7
            num_return = 1
            
        # 执行文本生成
        print("\n生成中...\n")
        result = pipe(
            prompt, 
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return,
            do_sample=True
        )
        
        # 显示结果
        print("-" * 80)
        for i, res in enumerate(result):
            print(f"生成结果 {i+1}:")
            print(res['generated_text'])
            print("-" * 80)

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于Transformers的文本生成")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"生成模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", help="提示文本")
    parser.add_argument("--max_length", type=int, default=50, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数(0.1-1.0)")
    parser.add_argument("--num_return", type=int, default=1, help="生成结果数量")
    args = parser.parse_args()
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建文本生成pipeline
    print(f"加载文本生成模型: {args.model}")
    pipe = create_pipeline(
        task="text-generation",
        model_name=args.model
    )
    
    # 如果命令行提供了提示文本，直接生成
    if args.prompt:
        result = pipe(
            args.prompt, 
            max_length=args.max_length,
            temperature=args.temperature,
            num_return_sequences=args.num_return,
            do_sample=True
        )
        
        print("\n生成结果:")
        print("-" * 80)
        for i, res in enumerate(result):
            print(f"生成结果 {i+1}:")
            print(res['generated_text'])
            print("-" * 80)
    else:
        # 否则进入交互模式
        interactive_generation(pipe)
    
    print("\n感谢使用文本生成系统！")

if __name__ == "__main__":
    main() 