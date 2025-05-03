#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
聊天机器人示例脚本
展示如何使用Transformers的conversation pipeline创建简单的聊天机器人
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import get_device, create_pipeline

# 默认会话模型
DEFAULT_MODEL = "facebook/blenderbot-400M-distill"

def interactive_chat(pipe):
    """交互式聊天"""
    print("\n欢迎使用聊天机器人！")
    print("输入'退出'或'exit'结束对话\n")
    
    # 初始化对话历史
    conversation_history = []
    
    while True:
        # 获取用户输入
        user_input = input("用户: ").strip()
        if user_input.lower() in ["退出", "exit"]:
            break
            
        # 如果是新对话
        if not conversation_history:
            conversation = pipe(user_input)
        else:
            # 将用户输入添加到现有对话
            conversation_history.append(user_input)
            conversation = pipe(conversation_history)
            
        # 获取模型响应
        bot_response = conversation.generated_responses[-1]
        conversation_history = conversation.to_dict()["past_user_inputs"]
        
        print(f"机器人: {bot_response}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于Transformers的聊天机器人")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"聊天模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--message", help="单条消息模式：输入一条消息并获取回复")
    args = parser.parse_args()
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建会话pipeline
    print(f"加载聊天模型: {args.model}")
    pipe = create_pipeline(
        task="conversational",
        model_name=args.model
    )
    
    # 如果命令行提供了消息，直接回复
    if args.message:
        conversation = pipe(args.message)
        bot_response = conversation.generated_responses[-1]
        print(f"\n用户: {args.message}")
        print(f"机器人: {bot_response}")
    else:
        # 否则进入交互模式
        interactive_chat(pipe)
    
    print("\n感谢使用聊天机器人！")

if __name__ == "__main__":
    main() 