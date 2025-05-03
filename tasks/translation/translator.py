#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
翻译示例脚本
展示如何使用Transformers的translation pipeline创建翻译应用
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import get_device, create_pipeline

# 默认翻译模型
TRANSLATION_MODELS = {
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",  # 英文到中文
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",  # 中文到英文
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",  # 英文到法文
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",  # 法文到英文
    "en-de": "Helsinki-NLP/opus-mt-en-de",  # 英文到德文
    "de-en": "Helsinki-NLP/opus-mt-de-en",  # 德文到英文
    "en-ja": "Helsinki-NLP/opus-mt-en-jap", # 英文到日文
    "ja-en": "Helsinki-NLP/opus-mt-jap-en"  # 日文到英文
}

def select_language_pair():
    """交互式选择语言对"""
    print("\n可选语言对:")
    print("-" * 50)
    for i, (key, model) in enumerate(TRANSLATION_MODELS.items()):
        print(f"{i+1}. {key}: {model}")
        
    while True:
        try:
            choice = int(input("\n请选择语言对 (1-8): ").strip())
            if 1 <= choice <= len(TRANSLATION_MODELS):
                return list(TRANSLATION_MODELS.items())[choice-1]
        except ValueError:
            pass
        print("无效选择，请重新输入")

def interactive_translation(pipe, lang_pair):
    """交互式翻译"""
    source_lang, target_lang = lang_pair.split("-")
    print(f"\n欢迎使用翻译系统！({source_lang} → {target_lang})")
    print("输入'退出'或'exit'结束使用\n")
    
    while True:
        # 获取用户输入
        text = input(f"{source_lang} > ").strip()
        if text.lower() in ["退出", "exit"]:
            break
            
        # 执行翻译
        result = pipe(text)
        
        # 显示结果
        translated_text = result[0]["translation_text"]
        print(f"{target_lang} > {translated_text}\n")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于Transformers的翻译系统")
    parser.add_argument("--lang_pair", choices=TRANSLATION_MODELS.keys(), help="语言对 (例如: en-zh)")
    parser.add_argument("--text", help="要翻译的文本")
    parser.add_argument("--model", help="指定翻译模型路径或名称")
    args = parser.parse_args()
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 选择语言对和模型
    if args.lang_pair:
        lang_pair = args.lang_pair
        model_name = args.model or TRANSLATION_MODELS[lang_pair]
    elif args.model:
        model_name = args.model
        # 尝试从模型名称中获取语言对
        for lp, model in TRANSLATION_MODELS.items():
            if model == model_name:
                lang_pair = lp
                break
        else:
            lang_pair = "unknown-unknown"
    else:
        # 交互式选择
        lang_pair, model_name = select_language_pair()
    
    # 创建翻译pipeline
    print(f"加载翻译模型: {model_name}")
    pipe = create_pipeline(
        task="translation",
        model_name=model_name
    )
    
    # 如果命令行提供了文本，直接翻译
    if args.text:
        result = pipe(args.text)
        translated_text = result[0]["translation_text"]
        
        source_lang, target_lang = lang_pair.split("-")
        print(f"\n{source_lang} > {args.text}")
        print(f"{target_lang} > {translated_text}")
    else:
        # 否则进入交互模式
        interactive_translation(pipe, lang_pair)
    
    print("\n感谢使用翻译系统！")

if __name__ == "__main__":
    main() 