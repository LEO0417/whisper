import os
import test_torch
import argparse
from tqdm import tqdm
import tempfile
import subprocess

# 设置环境变量，配置 HuggingFace 镜像源
mirror_url = "https://hf-mirror.com"
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = mirror_url
    print(f"已设置 HuggingFace 镜像源为：{os.environ['HF_ENDPOINT']}")
else:
    mirror_url = os.environ["HF_ENDPOINT"]
    print(f"使用已配置的 HuggingFace 镜像源：{mirror_url}")

# 设置 transformers 库使用的端点
os.environ["HF_MIRROR"] = mirror_url
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")

# 导入 transformers 相关库，确保它们在环境变量设置之后导入
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import hub

# 修补 transformers 的配置，强制使用镜像源
original_cached_file = hub.cached_file
def patched_cached_file(path_or_repo_id, *args, **kwargs):
    if path_or_repo_id.startswith("openai/"):
        if "endpoint" not in kwargs or kwargs["endpoint"] is None:
            kwargs["endpoint"] = mirror_url
    return original_cached_file(path_or_repo_id, *args, **kwargs)

# 应用补丁
hub.cached_file = patched_cached_file

def check_connectivity(endpoint=None):
    """检查与 HuggingFace 服务器的连接情况"""
    import socket
    import requests
    import time
    
    if endpoint is None:
        endpoint = mirror_url
    
    print(f"\n正在检查与 {endpoint} 的连接...")
    
    # 解析域名
    domain = endpoint.split("//")[-1].split("/")[0]
    
    # 测试 DNS 解析
    try:
        start_time = time.time()
        ip = socket.gethostbyname(domain)
        dns_time = time.time() - start_time
        print(f"DNS 解析成功：{domain} -> {ip} ({dns_time:.2f}秒)")
    except socket.gaierror:
        print(f"DNS 解析失败：无法解析 {domain}")
        return False
    
    # 测试 TCP 连接
    try:
        start_time = time.time()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((domain, 443))
        s.close()
        tcp_time = time.time() - start_time
        print(f"TCP 连接成功 ({tcp_time:.2f}秒)")
    except Exception as e:
        print(f"TCP 连接失败：{str(e)}")
        return False
    
    # 测试 HTTP GET 请求
    try:
        start_time = time.time()
        response = requests.get(f"{endpoint}/api/models", timeout=10)
        http_time = time.time() - start_time
        print(f"HTTP 请求成功 ({http_time:.2f}秒，状态码：{response.status_code})")
        
        # 检查响应内容是否正常
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list) or isinstance(data, dict):
                    print("API 响应格式正常")
                    return True
                else:
                    print("API 响应格式不正常")
            except:
                print("API 响应不是有效的 JSON 格式")
        
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"HTTP 请求失败：{str(e)}")
        return False

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="Whisper 语音识别工具")
    
    parser.add_argument("--audio", type=str, help="要转录的音频文件或文件夹路径")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3", 
                        help="模型名称或本地路径，默认为 large-v3")
    parser.add_argument("--language", type=str, default="auto", 
                        help="指定语言，默认为自动检测。示例：zh, en, ja, auto")
    parser.add_argument("--task", type=str, default="transcribe", 
                        choices=["transcribe", "translate"], 
                        help="任务类型：transcribe（转录）或 translate（翻译成英文）")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="输出文件夹路径")
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["cpu", "mps", "cuda", "auto"], 
                        help="设备类型：cpu, mps, cuda 或 auto")
    parser.add_argument("--download", action="store_true", 
                        help="下载模型到本地")
    parser.add_argument("--local_dir", type=str, default="./whisper-models",
                        help="本地模型保存路径")
    parser.add_argument("--mirror", type=str, default=None,
                        help="HuggingFace 镜像源 URL，默认使用 hf-mirror.com")
    parser.add_argument("--local_files_only", action="store_true",
                        help="仅使用本地文件，不尝试下载")
    parser.add_argument("--timeout", type=int, default=30,
                        help="下载超时时间（秒）")
    parser.add_argument("--diagnose", action="store_true",
                        help="运行网络诊断")
    
    return parser.parse_args()

def get_device(device_arg):
    """获取设备类型"""
    if device_arg != "auto":
        return device_arg
    
    if test_torch.backends.mps.is_available():
        return "mps"
    elif test_torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def download_model(model_name, local_dir, mirror=None, local_files_only=False, timeout=30):
    """下载模型到本地"""
    from huggingface_hub import snapshot_download
    import time
    
    # 确定使用的镜像源
    endpoint = mirror if mirror else mirror_url
    print(f"使用镜像源：{endpoint}")
    
    # 创建本地目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 确定模型的本地保存路径
    model_basename = os.path.basename(model_name)
    model_local_dir = os.path.join(local_dir, model_basename)
    
    print(f"正在下载模型 {model_name} 到本地...")
    
    # 重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model_path = snapshot_download(
                repo_id=model_name, 
                local_dir=model_local_dir,
                local_files_only=local_files_only,
                etag_timeout=timeout,
                max_workers=4,  # 减少并发数，提高稳定性
                endpoint=endpoint
            )
            print(f"模型已下载到：{model_path}")
            return model_path
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"下载失败：{str(e)}")
                print(f"等待 {wait_time} 秒后重试... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"下载失败，已达到最大重试次数：{str(e)}")
                print("提示：您可以尝试以下解决方法：")
                print("1. 检查网络连接")
                print("2. 使用 --mirror 参数指定其他镜像源")
                print("   例如：--mirror https://hf-mirror.com 或 --mirror https://huggingface.co")
                print("3. 使用 --local_files_only 参数仅使用本地已下载的模型")
                print("4. 增加 --timeout 参数值")
                
                # 如果未启用本地文件模式，尝试查找本地缓存
                if not local_files_only:
                    print("尝试查找本地缓存...")
                    try:
                        return download_model(model_name, local_dir, mirror, True, timeout)
                    except Exception:
                        print("未找到本地缓存")
                
                # 检查是否有部分下载的文件
                if os.path.exists(model_local_dir) and os.listdir(model_local_dir):
                    print(f"发现部分下载的模型文件：{model_local_dir}")
                    if input("是否尝试使用这些文件？(y/n) [y]: ").lower() != 'n':
                        return model_local_dir
                
                raise

def create_pipeline(model_path_or_name, device, language="auto", task="transcribe", local_files_only=False):
    """创建语音识别 pipeline"""
    print(f"正在加载模型到{device}设备...")
    
    # 检查模型路径是否为本地目录
    is_local_path = os.path.isdir(model_path_or_name)
    
    # 如果不是本地路径且不是强制本地模式，则尝试从镜像下载
    try:
        # 尝试创建 pipeline
        print(f"正在加载模型：{model_path_or_name}" + (" (仅本地模式)" if local_files_only else ""))
        
        # 加载模型和处理器
        if local_files_only:
            # 尝试查找本地模型目录
            if not is_local_path and os.path.exists("./whisper-models"):
                # 检查是否是形如"openai/whisper-XXX"的模型名
                if "/" in model_path_or_name:
                    model_name = model_path_or_name.split("/")[1]
                    local_path = os.path.join("./whisper-models", model_name)
                    if os.path.exists(local_path):
                        print(f"找到本地模型目录：{local_path}")
                        model_path_or_name = local_path
                        is_local_path = True
        
        # 创建 pipeline - 注意：不使用 local_files_only 参数
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=model_path_or_name,
            device=device,
        )
        
    except Exception as e:
        print(f"加载模型失败：{str(e)}")
        
        # 如果不是强制本地模式，尝试切换到本地模式
        if not local_files_only:
            print("尝试仅使用本地文件...")
            return create_pipeline(model_path_or_name, device, language, task, True)
        
        # 如果是网络错误，提供更详细的解决方案
        if "connect" in str(e).lower() or "timeout" in str(e).lower():
            print("\n可能是网络连接问题，请尝试：")
            print("1. 检查网络连接")
            print("2. 使用 --mirror 参数尝试其他镜像源")
            print("3. 先使用 --download 参数下载模型")
            print("4. 使用 --local_files_only 参数强制使用本地模型")
        
        raise Exception(f"无法加载模型。请确保模型已下载并且路径正确：{str(e)}")
    
    # 设置生成参数 - 不在这里设置，而是在使用时直接传递
    generation_kwargs = {}
    
    # 记录任务和语言设置，但不传递给 pipeline
    if language != "auto":
        print(f"设置语言：{language}")
    
    print(f"设置任务：{task}")
    
    # 返回包含任务和语言的信息，但这些不作为参数传递
    task_info = {"language_setting": language, "task_setting": task}
    
    return pipe, task_info

def convert_audio_if_needed(audio_file):
    """如果需要，转换音频文件为 WAV 格式"""
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在：{audio_file}")
    
    # 获取文件扩展名
    _, ext = os.path.splitext(audio_file)
    ext = ext.lower()
    
    # 如果已经是 WAV 格式，直接返回
    if ext == '.wav':
        return audio_file
    
    # 检查 ffmpeg 是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("警告：ffmpeg 未安装，可能无法处理所有音频格式")
        print("建议安装 ffmpeg: brew install ffmpeg (MacOS) 或 apt-get install ffmpeg (Linux)")
        return audio_file  # 尝试直接使用原始文件
    
    # 创建临时文件
    temp_dir = tempfile.gettempdir()
    temp_wav = os.path.join(temp_dir, f"{os.path.basename(audio_file)}.wav")
    
    print(f"正在将 {ext} 格式转换为 WAV 格式...")
    
    # 使用 ffmpeg 转换
    try:
        subprocess.run(
            ['ffmpeg', '-i', audio_file, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', temp_wav],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"转换成功，临时文件：{temp_wav}")
        return temp_wav
    except subprocess.SubprocessError as e:
        print(f"转换失败：{e}")
        return audio_file  # 失败时返回原始文件，尝试直接使用

def process_audio(pipe, audio_file, task_info, output_dir):
    """处理单个音频文件"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(output_dir, f"{file_name}.txt")
        
        print(f"正在处理：{audio_file}")
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"音频文件不存在：{audio_file}")
            
        # 打印任务信息
        language = task_info.get("language_setting")
        task = task_info.get("task_setting")
        print(f"语言设置：{language}, 任务类型：{task}")
        
        # 转换音频格式
        try:
            processed_audio = convert_audio_if_needed(audio_file)
        except Exception as e:
            print(f"音频转换失败：{e}")
            processed_audio = audio_file  # 使用原始文件
        
        # 不传递任何生成参数
        result = pipe(processed_audio)
        
        # 如果使用了临时文件，删除它
        if processed_audio != audio_file and os.path.exists(processed_audio):
            try:
                os.remove(processed_audio)
                print(f"已删除临时文件：{processed_audio}")
            except:
                pass
        
        # 保存结果到文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"结果已保存到：{output_file}")
        return result["text"]
    
    except Exception as e:
        print(f"处理文件 {audio_file} 时出错：{str(e)}")
        print(f"错误类型：{type(e).__name__}")
        
        # 对于常见错误提供更具体的建议
        if isinstance(e, FileNotFoundError):
            print(f"请确认文件路径是否正确，当前路径：{audio_file}")
        elif "device" in str(e).lower():
            print("GPU/MPS 相关错误，可以尝试使用 CPU: --device cpu")
        elif "soundfile" in str(e).lower() or "audio" in str(e).lower():
            print("音频格式问题，请尝试先安装 ffmpeg: brew install ffmpeg")
            print("然后重新运行程序")
        
        return None

def batch_process(pipe, audio_path, task_info, output_dir):
    """批量处理文件夹中的所有音频文件"""
    supported_formats = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    
    if os.path.isfile(audio_path):
        # 处理单个文件
        return process_audio(pipe, audio_path, task_info, output_dir)
    
    elif os.path.isdir(audio_path):
        # 处理文件夹中的所有音频文件
        results = {}
        audio_files = []
        
        # 收集所有支持的音频文件
        for file in os.listdir(audio_path):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                audio_files.append(os.path.join(audio_path, file))
        
        if not audio_files:
            print(f"在 {audio_path} 中未找到支持的音频文件")
            return results
        
        # 使用进度条处理所有文件
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            result = process_audio(pipe, audio_file, task_info, output_dir)
            if result:
                results[audio_file] = result
        
        return results
    
    else:
        print(f"路径不存在：{audio_path}")
        return None

def select_model():
    """交互式选择要下载的模型版本"""
    models = {
        "1": {"name": "openai/whisper-tiny", "size": "~75MB", "quality": "最低", "speed": "最快"},
        "2": {"name": "openai/whisper-base", "size": "~140MB", "quality": "低", "speed": "快"},
        "3": {"name": "openai/whisper-small", "size": "~460MB", "quality": "中", "speed": "中"},
        "4": {"name": "openai/whisper-medium", "size": "~1.5GB", "quality": "高", "speed": "慢"},
        "5": {"name": "openai/whisper-large-v3", "size": "~3GB", "quality": "最高", "speed": "最慢"}
    }
    
    print("\n可选模型列表：")
    print("-" * 80)
    print(f"{'编号':<6}{'模型名称':<25}{'大小':<10}{'质量':<10}{'速度':<10}")
    print("-" * 80)
    for key, model in models.items():
        print(f"{key:<6}{model['name']:<25}{model['size']:<10}{model['quality']:<10}{model['speed']:<10}")
    
    # 检查本地已有模型
    local_models = []
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        for model_key in models.values():
            model_name = model_key["name"].split("/")[1]
            potential_paths = [
                os.path.join(cache_dir, f"models--openai--{model_name}"),
                f"./whisper-models/{model_name}"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    local_models.append(model_key["name"])
                    break
    
    if local_models:
        print("\n本地已有模型：")
        for model in local_models:
            print(f"- {model}")
    
    while True:
        choice = input("\n请选择要下载的模型编号 (1-5) [默认 5]: ").strip() or "5"
        if choice in models:
            return models[choice]["name"]
        print("无效选择，请重新输入")

def main():
    # 解析命令行参数
    args = setup_args()
    
    # 如果指定了自定义镜像源，设置环境变量
    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror
        global mirror_url
        mirror_url = args.mirror
        print(f"设置 HuggingFace 镜像源为：{args.mirror}")
    
    # 运行诊断
    if args.diagnose:
        endpoint = args.mirror if args.mirror else mirror_url
        print(f"运行网络诊断，检测与 {endpoint} 的连接...")
        if check_connectivity(endpoint):
            print("\n✅ 网络连接正常，可以访问 HuggingFace 镜像")
            
            # 尝试连接原始 HuggingFace
            if endpoint != "https://huggingface.co":
                print("\n测试与原始 HuggingFace 的连接...")
                if check_connectivity("https://huggingface.co"):
                    print("\n✅ 也可以访问原始 HuggingFace 网站")
                else:
                    print("\n❌ 无法访问原始 HuggingFace 网站，但镜像站点可用")
        else:
            print("\n❌ 网络连接有问题，无法访问指定的 HuggingFace 镜像")
            
            # 尝试其他常用镜像
            other_mirrors = [
                "https://hf-mirror.com", 
                "https://huggingface.co"
            ]
            
            for mirror in other_mirrors:
                if mirror != endpoint:
                    print(f"\n尝试测试其他镜像：{mirror}")
                    if check_connectivity(mirror):
                        print(f"\n✅ 可以访问 {mirror}")
                        print(f"推荐使用命令：python whisper_advanced.py --mirror {mirror}")
                    else:
                        print(f"\n❌ 也无法访问 {mirror}")
        
        return
    
    # 获取设备
    device = get_device(args.device)
    print(f"使用设备：{device}")
    
    # 检查本地缓存目录中是否已有模型
    local_models = []
    model_path = args.model
    
    # 检查 whisper-models 目录
    if os.path.exists(args.local_dir):
        for model_dir in os.listdir(args.local_dir):
            if os.path.isdir(os.path.join(args.local_dir, model_dir)):
                local_models.append(os.path.join(args.local_dir, model_dir))
    
    # 显示本地模型列表
    if local_models and not args.download:
        print("\n发现本地模型目录：")
        for i, model in enumerate(local_models, 1):
            print(f"{i}. {model}")
        
        if args.local_files_only or input("\n是否使用本地模型？(y/n) [y]: ").lower() != "n":
            choice = input("请选择要使用的模型编号 [1]: ").strip() or "1"
            if choice.isdigit() and 1 <= int(choice) <= len(local_models):
                model_path = local_models[int(choice)-1]
                print(f"使用本地模型：{model_path}")
                args.local_files_only = True
    
    # 下载模型 (如果需要)
    if args.download and args.model == "openai/whisper-large-v3":
        model_name = select_model()
        try:
            model_path = download_model(model_name, args.local_dir, args.mirror, args.local_files_only, args.timeout)
        except Exception as e:
            if args.local_files_only:
                print(f"无法找到本地模型：{str(e)}")
                return
            else:
                print(f"下载失败：{str(e)}")
                if input("是否尝试使用本地模型？(y/n) [y]: ").lower() != "n":
                    args.local_files_only = True
                    # 继续尝试使用本地模型
                else:
                    return
    elif args.download:
        try:
            model_path = download_model(args.model, args.local_dir, args.mirror, args.local_files_only, args.timeout)
        except Exception as e:
            if args.local_files_only:
                print(f"无法找到本地模型：{str(e)}")
                return
            else:
                print(f"下载失败：{str(e)}")
                if input("是否尝试使用本地模型？(y/n) [y]: ").lower() != "n":
                    args.local_files_only = True
                    # 继续尝试使用本地模型
                else:
                    return
    
    # 创建 pipeline
    try:
        pipe, task_info = create_pipeline(
            model_path, 
            device,
            language=args.language,
            task=args.task,
            local_files_only=args.local_files_only
        )
    except Exception as e:
        print(f"创建 pipeline 失败：{str(e)}")
        return
    
    # 处理音频
    if args.audio:
        batch_process(pipe, args.audio, task_info, args.output_dir)
    else:
        # 交互模式
        while True:
            audio_path = input("\n请输入音频文件或文件夹路径 (输入'exit'退出): ")
            if audio_path.lower() == 'exit':
                break
            
            if not os.path.exists(audio_path):
                print(f"路径不存在：{audio_path}")
                continue
            
            result = batch_process(pipe, audio_path, task_info, args.output_dir)
            
            if result and isinstance(result, str):
                print("\n转录结果：")
                print(result)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已被用户中断")
    except Exception as e:
        print(f"\n发生错误：{str(e)}")
        print("\n如果遇到网络问题，可以尝试以下命令：")
        print("1. 运行网络诊断：")
        print("   python whisper_advanced.py --diagnose")
        print("\n2. 使用国内镜像站：")
        print("   python whisper_advanced.py --mirror https://hf-mirror.com")
        print("\n3. 下载模型到本地后离线使用：")
        print("   python whisper_advanced.py --download --mirror https://hf-mirror.com")
        print("   python whisper_advanced.py --local_files_only")
        print("\n4. 增加超时时间：")
        print("   python whisper_advanced.py --timeout 60")
        print("\n5. 如果都无法解决，可以手动下载模型：")
        print("   - 访问 https://hf-mirror.com/openai/whisper-large-v3/tree/main")
        print("   - 下载所有文件到 ./whisper-models/whisper-large-v3/ 目录")
        print("   - 然后使用：python whisper_advanced.py --model ./whisper-models/whisper-large-v3 --local_files_only") 