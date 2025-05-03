#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
问答系统示例脚本
展示如何使用Transformers的问答pipeline完成基于上下文的问答任务
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import get_device, create_pipeline

# 默认问答模型
DEFAULT_MODEL = "distilbert-base-cased-distilled-squad"

def interactive_qa(pipe):
    """交互式问答"""
    print("\n欢迎使用问答系统！")
    print("输入'退出'或'exit'结束对话\n")
    
    while True:
        # 获取上下文
        print("-" * 80)
        context = input("请输入上下文文本: ").strip()
        if context.lower() in ["退出", "exit"]:
            break
            
        # 获取问题
        question = input("请输入问题: ").strip()
        if question.lower() in ["退出", "exit"]:
            break
            
        # 执行问答
        print("\n思考中...\n")
        result = pipe(question=question, context=context)
        
        # 显示结果
        print(f"问题: {question}")
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['score']:.4f}")
        print("-" * 80 + "\n")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于Transformers的问答系统")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"问答模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--context", help="上下文文本")
    parser.add_argument("--question", help="问题文本")
    args = parser.parse_args()
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建问答pipeline
    print(f"加载问答模型: {args.model}")
    pipe = create_pipeline(
        task="question-answering",
        model_name=args.model
    )
    
    # 如果命令行提供了上下文和问题，直接回答
    if args.context and args.question:
        result = pipe(question=args.question, context=args.context)
        print(f"\n问题: {args.question}")
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['score']:.4f}")
    else:
        # 否则进入交互模式
        interactive_qa(pipe)
    
    print("\n感谢使用问答系统！")

if __name__ == "__main__":
    main() 