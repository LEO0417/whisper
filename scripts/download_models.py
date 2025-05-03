#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型下载脚本
用于下载和缓存常用的Transformers模型
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import download_model, list_local_models

# 常用模型列表
COMMON_MODELS = {
    "speech_recognition": [
        {"name": "openai/whisper-tiny", "size": "~75MB"},
        {"name": "openai/whisper-base", "size": "~140MB"},
        {"name": "openai/whisper-small", "size": "~460MB"},
        {"name": "openai/whisper-medium", "size": "~1.5GB"},
        {"name": "openai/whisper-large-v3", "size": "~3GB"}
    ],
    "question_answering": [
        {"name": "distilbert-base-cased-distilled-squad", "size": "~250MB"}
    ],
    "conversation": [
        {"name": "facebook/blenderbot-400M-distill", "size": "~650MB"}
    ],
    "text_generation": [
        {"name": "gpt2", "size": "~500MB"},
        {"name": "gpt2-medium", "size": "~1.5GB"}
    ],
    "translation": [
        {"name": "Helsinki-NLP/opus-mt-en-zh", "size": "~300MB"},
        {"name": "Helsinki-NLP/opus-mt-zh-en", "size": "~300MB"}
    ]
}

def select_models_to_download():
    """交互式选择要下载的模型"""
    selected_models = []
    
    print("\n=== 可下载的模型列表 ===\n")
    
    # 显示已有本地模型
    local_models = list_local_models()
    if local_models:
        print("本地已有模型:")
        for model in local_models:
            print(f"✓ {model}")
        print()
    
    # 显示可选模型
    for i, (task, models) in enumerate(COMMON_MODELS.items()):
        print(f"\n{i+1}. {task} 任务:")
        for j, model in enumerate(models):
            model_id = f"{i+1}.{j+1}"
            is_local = model["name"] in local_models
            status = "✓" if is_local else " "
            print(f"  [{status}] {model_id}: {model['name']} ({model['size']})")
    
    # 选择模型
    print("\n请选择要下载的模型:")
    print("  - 输入模型编号 (例如: '1.1' 表示语音识别中的第一个模型)")
    print("  - 输入任务编号 (例如: '1' 表示下载所有语音识别模型)")
    print("  - 输入 'all' 下载所有模型")
    print("  - 输入多个编号，用逗号分隔 (例如: '1.1,2.1,3.1')")
    print("  - 输入 'q' 退出")
    
    while True:
        choice = input("\n请选择: ").strip().lower()
        
        if choice == 'q':
            break
            
        if choice == 'all':
            for task_models in COMMON_MODELS.values():
                for model in task_models:
                    selected_models.append(model["name"])
            break
            
        # 处理输入
        for item in choice.split(','):
            item = item.strip()
            
            # 检查是否为任务编号
            if item.isdigit():
                task_idx = int(item) - 1
                if 0 <= task_idx < len(COMMON_MODELS):
                    task_name = list(COMMON_MODELS.keys())[task_idx]
                    for model in COMMON_MODELS[task_name]:
                        selected_models.append(model["name"])
                else:
                    print(f"无效的任务编号: {item}")
                    
            # 检查是否为模型编号 (例如: 1.2)
            elif '.' in item:
                try:
                    task_idx, model_idx = map(int, item.split('.'))
                    task_idx -= 1
                    model_idx -= 1
                    
                    if 0 <= task_idx < len(COMMON_MODELS):
                        task_name = list(COMMON_MODELS.keys())[task_idx]
                        if 0 <= model_idx < len(COMMON_MODELS[task_name]):
                            selected_models.append(COMMON_MODELS[task_name][model_idx]["name"])
                        else:
                            print(f"无效的模型编号: {item}")
                    else:
                        print(f"无效的任务编号: {item}")
                except ValueError:
                    print(f"无效的编号格式: {item}")
    
    # 去重
    return list(set(selected_models))

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="下载和缓存常用的Transformers模型")
    parser.add_argument("--model", help="指定要下载的模型名称")
    parser.add_argument("--task", choices=COMMON_MODELS.keys(), help="下载特定任务的所有模型")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--local_dir", help="本地保存目录，默认为 './models'")
    args = parser.parse_args()
    
    # 设置本地保存目录
    local_dir = args.local_dir or "./models"
    
    # 确定要下载的模型列表
    models_to_download = []
    
    if args.model:
        # 下载指定模型
        models_to_download.append(args.model)
    elif args.task:
        # 下载特定任务的所有模型
        for model in COMMON_MODELS[args.task]:
            models_to_download.append(model["name"])
    elif args.all:
        # 下载所有模型
        for task_models in COMMON_MODELS.values():
            for model in task_models:
                models_to_download.append(model["name"])
    else:
        # 交互式选择
        models_to_download = select_models_to_download()
    
    # 开始下载
    if not models_to_download:
        print("未选择任何模型，退出")
        return
        
    print(f"\n将下载 {len(models_to_download)} 个模型到 {local_dir}:")
    for model in models_to_download:
        print(f"- {model}")
        
    if input("\n确认下载? (y/n): ").strip().lower() != 'y':
        print("取消下载")
        return
        
    # 执行下载
    for i, model in enumerate(models_to_download):
        print(f"\n[{i+1}/{len(models_to_download)}] 下载模型: {model}")
        task_type = None
        for task, models in COMMON_MODELS.items():
            if any(m["name"] == model for m in models):
                task_type = task
                break
                
        # 根据任务类型设置保存路径
        if task_type:
            model_local_dir = os.path.join(local_dir, task_type, model.split('/')[-1])
        else:
            model_local_dir = os.path.join(local_dir, model.split('/')[-1])
            
        try:
            download_model(model, local_dir=model_local_dir)
            print(f"模型 {model} 下载完成")
        except Exception as e:
            print(f"下载模型 {model} 失败: {str(e)}")
    
    print("\n所有模型下载完成")

if __name__ == "__main__":
    main() 