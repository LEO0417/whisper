#!/usr/bin/env python
# 下载所有尺寸的Whisper模型

import os
import subprocess
import time
import argparse
import shutil
import datetime
import threading
import sys

# 尝试导入彩色输出库，如果不可用则使用模拟函数
try:
    from colorama import init, Fore, Style
    init()  # 初始化colorama
    def green(text): return Fore.GREEN + text + Style.RESET_ALL
    def red(text): return Fore.RED + text + Style.RESET_ALL
    def yellow(text): return Fore.YELLOW + text + Style.RESET_ALL
    def blue(text): return Fore.BLUE + text + Style.RESET_ALL
    def cyan(text): return Fore.CYAN + text + Style.RESET_ALL
    def magenta(text): return Fore.MAGENTA + text + Style.RESET_ALL
except ImportError:
    # 如果没有colorama库，使用普通文本
    def green(text): return text
    def red(text): return text
    def yellow(text): return text
    def blue(text): return text
    def cyan(text): return text
    def magenta(text): return text

# 模型大小估计（单位：MB）
MODEL_SIZES = {
    "tiny": 75,
    "base": 140,
    "small": 460,
    "medium": 1500,
    "large-v2": 2900,
    "large-v3": 3000
}

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="下载所有尺寸的Whisper模型")
    parser.add_argument("--mirror", type=str, default="https://hf-mirror.com",
                      help="HuggingFace镜像源URL，默认使用hf-mirror.com")
    parser.add_argument("--local_dir", type=str, default="./whisper-models",
                      help="本地模型保存路径")
    parser.add_argument("--timeout", type=int, default=60,
                      help="下载超时时间（秒）")
    parser.add_argument("--activity_timeout", type=int, default=120,
                      help="下载过程中无活动超时时间（秒），用于检测卡住的下载")
    parser.add_argument("--retry", type=int, default=3,
                      help="失败后重试次数")
    parser.add_argument("--skip_existing", action="store_true",
                      help="跳过已存在的模型")
    parser.add_argument("--only", type=str, default=None,
                      help="只下载指定大小的模型（tiny, base, small, medium, large-v2, large-v3）")
    parser.add_argument("--check_disk", action="store_true",
                      help="下载前检查磁盘空间")
    parser.add_argument("--no_confirm", action="store_true",
                      help="无需确认直接下载")
    parser.add_argument("--parallel", action="store_true",
                      help="并行下载模型（可能导致网络拥塞）")
    parser.add_argument("--clean", action="store_true",
                      help="下载前清除可能存在的不完整模型文件")
    parser.add_argument("--alternative_mirrors", nargs='+', default=["https://hf-mirror.com", "https://huggingface.co"],
                      help="备选镜像列表，如果主镜像失败，将尝试这些镜像")
    return parser.parse_args()

def check_disk_space(path, required_mb):
    """检查指定路径的可用磁盘空间"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_mb = free / (1024 * 1024)  # 转换为MB
        
        print(f"当前可用磁盘空间: {green(f'{free_mb:.2f} MB')}")
        print(f"估计需要的空间: {yellow(f'{required_mb:.2f} MB')}")
        
        if free_mb < required_mb:
            print(red(f"警告: 磁盘空间可能不足! 建议至少有 {required_mb * 1.1:.2f} MB 可用空间"))
            return False
        else:
            print(green(f"磁盘空间充足，可以继续下载"))
            return True
    except Exception as e:
        print(red(f"检查磁盘空间出错: {e}"))
        return True  # 如果无法检查，则假设有足够空间

def estimate_download_time(size_mb, speed_mbps=2):
    """估计下载时间，默认假设2MB/s的下载速度"""
    seconds = size_mb / speed_mbps
    time_str = str(datetime.timedelta(seconds=seconds))
    return time_str

def download_model(model_size, args, results=None):
    """下载指定大小的模型"""
    model_name = f"openai/whisper-{model_size}"
    local_path = os.path.join(args.local_dir, f"whisper-{model_size}")
    
    # 检查模型是否已存在
    if args.skip_existing and os.path.exists(local_path):
        if os.listdir(local_path):  # 确保目录不是空的
            print(green(f"模型 {model_name} 已存在于 {local_path}，跳过下载"))
            if results is not None:
                results[model_size] = True
            return True
    
    # 如果指定了清理参数且目录存在，删除可能不完整的文件
    if args.clean and os.path.exists(local_path):
        print(yellow(f"清理可能不完整的模型文件: {local_path}"))
        try:
            shutil.rmtree(local_path)
            print(green(f"成功清理目录: {local_path}"))
        except Exception as e:
            print(red(f"清理目录失败: {e}"))
    
    # 下载前显示预计大小和时间
    estimated_size = MODEL_SIZES.get(model_size, 0)
    if estimated_size > 0:
        print(f"预计模型大小: {cyan(f'{estimated_size} MB')}")
        print(f"预计下载时间: {cyan(estimate_download_time(estimated_size))}")
    
    # 尝试所有可能的镜像
    mirrors_to_try = [args.mirror] + [m for m in args.alternative_mirrors if m != args.mirror]
    
    # 记录开始时间
    start_time = time.time()
    
    # 对每个镜像尝试下载
    for mirror_index, mirror in enumerate(mirrors_to_try):
        print(cyan(f"使用镜像 {mirror_index+1}/{len(mirrors_to_try)}: {mirror}"))
        
        # 尝试下载，支持重试
        for attempt in range(args.retry):
            # 设置下载命令
            cmd = [
                "python", "whisper_advanced.py", 
                "--download", 
                "--model", model_name,
                "--mirror", mirror,
                "--local_dir", args.local_dir,
                "--timeout", str(args.timeout),
                "--no_confirm"  # 避免在子进程中请求输入
            ]
            
            try:
                print(cyan(f"正在下载 {model_name}，尝试 #{attempt+1}/{args.retry}，镜像: {mirror}"))
                
                # 使用Popen而不是run，以便实时输出
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1  # 行缓冲，确保实时输出
                )
                
                # 监控子进程输出并实时显示
                download_started = False
                download_finished = False
                last_output_time = time.time()
                
                while True:
                    # 检查进程是否仍在运行
                    if process.poll() is not None:
                        break
                    
                    # 读取输出行
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(stdout_line.strip())
                        last_output_time = time.time()
                        if "正在下载" in stdout_line:
                            download_started = True
                        if "模型已下载" in stdout_line:
                            download_finished = True
                    
                    # 读取错误行
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        print(red(stderr_line.strip()))
                        last_output_time = time.time()
                    
                    # 检查是否长时间没有输出（可能卡住）
                    if time.time() - last_output_time > args.activity_timeout and download_started and not download_finished:
                        print(red(f"下载似乎卡住了，超过{args.activity_timeout}秒没有输出"))
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                        raise TimeoutError("下载过程卡住")
                    
                    # 避免CPU占用过高
                    time.sleep(0.1)
                
                # 检查进程退出码
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
                
                # 验证模型是否真的下载成功
                if not os.path.exists(local_path) or not os.listdir(local_path):
                    raise FileNotFoundError(f"下载似乎完成，但找不到模型文件: {local_path}")
                
                # 计算下载用时
                elapsed_time = time.time() - start_time
                time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
                
                print(green(f"成功下载 {model_name}! 耗时: {time_str}"))
                if results is not None:
                    results[model_size] = True
                return True
                
            except (subprocess.CalledProcessError, TimeoutError, FileNotFoundError) as e:
                print(red(f"使用镜像 {mirror} 下载 {model_name} 失败: {e}"))
                
                # 如果当前镜像的尝试次数未达上限，则重试
                if attempt < args.retry - 1:
                    wait_time = 10 * (attempt + 1)  # 递增等待时间
                    print(yellow(f"等待 {wait_time} 秒后重试..."))
                    time.sleep(wait_time)
                else:
                    print(yellow(f"在镜像 {mirror} 上达到最大重试次数，尝试下一个镜像"))
                    break  # 尝试下一个镜像
    
    # 所有镜像都失败
    print(red(f"所有镜像尝试下载 {model_name} 均失败"))
    if results is not None:
        results[model_size] = False
    return False

def print_header():
    """打印脚本标题"""
    header = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   Whisper 模型下载工具                                   ║
    ║   -------------------------------------------           ║
    ║   支持自动下载多种尺寸的 Whisper 语音识别模型            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(cyan(header))

def print_progress(current, total, model_size, start_time):
    """打印简单的进度指示器"""
    elapsed = time.time() - start_time
    if elapsed < 1:
        return

    percent = current / total * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # 计算剩余时间
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0
    
    sys.stdout.write(f'\r模型 {model_size}: [{bar}] {percent:.1f}% ')
    sys.stdout.write(f'剩余时间: {str(datetime.timedelta(seconds=int(remaining)))}')
    sys.stdout.flush()

def main():
    """主函数"""
    print_header()
    args = setup_args()
    
    # 保存之前运行的结果文件路径
    results_file = os.path.join(args.local_dir, "download_results.txt")
    
    # 所有可用的模型大小
    all_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    
    # 要下载的模型列表
    if args.only:
        if args.only in all_models:
            models_to_download = [args.only]
        else:
            print(red(f"错误: 未知的模型大小 '{args.only}'"))
            print(yellow(f"可用的模型大小: {', '.join(all_models)}"))
            return
    else:
        models_to_download = all_models
    
    # 检查是否有之前未完成的下载
    previous_results = {}
    if os.path.exists(results_file) and not args.clean:
        try:
            with open(results_file, "r") as f:
                for line in f:
                    if ":" in line:
                        model, status = line.strip().split(":", 1)
                        previous_results[model] = status.strip() == "成功"
            
            if previous_results:
                print(yellow("发现之前的下载记录:"))
                for model, success in previous_results.items():
                    status = green("✓") if success else red("✗")
                    print(f"{status} {model}")
                
                if not args.no_confirm:
                    resume = input(cyan("是否继续之前未完成的下载? (y/n): ")).lower()
                    if resume == 'y':
                        # 只下载之前失败的模型
                        new_models = []
                        for model in models_to_download:
                            if model not in previous_results or not previous_results[model]:
                                new_models.append(model)
                        if new_models:
                            models_to_download = new_models
                            print(yellow(f"将只下载之前失败的模型: {', '.join(models_to_download)}"))
                        else:
                            print(green("所有模型已下载成功!"))
                            return
                    else:
                        # 删除结果文件，重新开始
                        os.remove(results_file)
                        previous_results = {}
        except Exception as e:
            print(red(f"读取之前的下载记录失败: {e}"))
            previous_results = {}
    
    # 确保目标目录存在
    os.makedirs(args.local_dir, exist_ok=True)
    
    # 计算需要的总空间
    total_size_mb = sum(MODEL_SIZES.get(model, 0) for model in models_to_download)
    
    # 显示下载计划
    print(f"将下载以下模型: {', '.join([cyan(f'whisper-{m}') for m in models_to_download])}")
    print(f"下载镜像: {blue(args.mirror)}")
    print(f"本地保存路径: {blue(os.path.abspath(args.local_dir))}")
    print(f"超时设置: {args.timeout}秒")
    print(f"活动超时: {args.activity_timeout}秒")
    print(f"重试次数: {args.retry}")
    print(f"总计大小: 约 {yellow(f'{total_size_mb} MB')} ({yellow(f'{total_size_mb/1024:.2f} GB')})")
    print(f"预计总下载时间: {yellow(estimate_download_time(total_size_mb))}")
    
    if args.parallel:
        print(yellow("注意：已启用并行下载，可能导致网络拥塞或下载失败"))
    
    # 检查磁盘空间
    if args.check_disk:
        if not check_disk_space(args.local_dir, total_size_mb * 1.2):  # 增加20%的缓冲
            if not args.no_confirm and input(yellow("磁盘空间可能不足，是否仍要继续? (y/n): ")).lower() != 'y':
                print("取消下载")
                return
    
    # 确认下载
    if not args.no_confirm and input(cyan("确认下载? (y/n): ")).lower() != 'y':
        print("取消下载")
        return
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 下载模型
    results = {}
    
    # 合并之前的结果
    for model, success in previous_results.items():
        if success and model not in models_to_download:
            results[model] = success
    
    if args.parallel:
        # 并行下载
        threads = []
        
        for model_size in models_to_download:
            print(f"\n{yellow('='*50)}")
            print(yellow(f"开始下载 whisper-{model_size}"))
            print(yellow(f"{'='*50}"))
            
            thread = threading.Thread(
                target=download_model, 
                args=(model_size, args, results)
            )
            thread.start()
            threads.append((model_size, thread))
            time.sleep(2)  # 短暂延迟，避免同时启动所有下载
        
        # 等待所有线程完成
        for model_size, thread in threads:
            thread.join()
            
    else:
        # 顺序下载
        for model_size in models_to_download:
            print(f"\n{yellow('='*50)}")
            print(yellow(f"开始下载 whisper-{model_size}"))
            print(yellow(f"{'='*50}"))
            
            success = download_model(model_size, args)
            results[model_size] = success
            
            # 每下载完一个模型，保存结果，以便恢复
            try:
                with open(results_file, "w") as f:
                    for model, success in results.items():
                        f.write(f"{model}: {'成功' if success else '失败'}\n")
            except Exception as e:
                print(red(f"保存下载记录失败: {e}"))
    
    # 计算总用时
    total_elapsed_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_elapsed_time)))
    
    # 打印下载结果摘要
    print("\n\n" + magenta("下载结果摘要:"))
    print(magenta("="*30))
    print(f"总耗时: {cyan(total_time_str)}")
    for model_size, success in results.items():
        status = green("✅ 成功") if success else red("❌ 失败")
        model_size_str = cyan(f"whisper-{model_size}")
        size_str = f"({MODEL_SIZES.get(model_size, 0)} MB)"
        print(f"{model_size_str} {size_str}: {status}")
    
    # 保存最终结果
    try:
        with open(results_file, "w") as f:
            for model, success in results.items():
                f.write(f"{model}: {'成功' if success else '失败'}\n")
    except Exception as e:
        print(red(f"保存最终下载记录失败: {e}"))
    
    # 计算已下载的总大小
    successful_models = [model for model, success in results.items() if success]
    total_downloaded_mb = sum(MODEL_SIZES.get(model, 0) for model in successful_models)
    print(f"已成功下载: {green(f'{len(successful_models)}/{len(results)}')} 个模型，总计约 {green(f'{total_downloaded_mb} MB')} ({green(f'{total_downloaded_mb/1024:.2f} GB')})")
    
    # 检查是否有失败的下载
    failed = [model for model, success in results.items() if not success]
    if failed:
        print(red(f"\n以下模型下载失败: {', '.join(['whisper-'+m for m in failed])}"))
        print(yellow("可以使用以下命令重试下载失败的模型:"))
        for model in failed:
            print(cyan(f"python download_all_models.py --only {model} --mirror {args.mirror} --clean"))
    else:
        print(green("\n所有模型下载成功!"))
        # 下载完成后删除结果文件
        if os.path.exists(results_file):
            try:
                os.remove(results_file)
            except:
                pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(red("\n程序被用户中断"))
        sys.exit(1)
    except Exception as e:
        print(red(f"\n程序出错: {e}"))
        sys.exit(1) 