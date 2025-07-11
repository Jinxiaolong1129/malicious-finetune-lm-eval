#!/usr/bin/env python3
"""
LM-Eval 数据下载脚本
在本地联网环境中运行 lm-eval 来下载所有需要的数据集
然后可以将缓存目录传输到离线服务器
"""

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import subprocess
import sys
from pathlib import Path
import shutil
import time


CACHE_DIR = "/data3/user/jin509/new_hf_cache"


def setup_environment():
    """设置环境变量"""
    # 设置 HuggingFace 缓存目录
    os.environ["HF_HOME"] = CACHE_DIR
    
    # 创建缓存目录
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 设置缓存目录: {CACHE_DIR}")
    return CACHE_DIR

def download_datasets_via_lm_eval():
    """使用 lm-eval 下载数据集"""
    
    # 需要下载的评估任务列表
    task_defaults = {
        "mmlu": {"num_fewshot": 0, "batch_size": "auto"},
        "humaneval": {"num_fewshot": 0, "batch_size": "auto"},
        "gsm8k": {"num_fewshot": 0, "batch_size": "auto"},
        "arc_challenge": {"num_fewshot": None, "batch_size": "auto"},  # 使用默认值
        "truthfulqa_mc1": {"num_fewshot": 0, "batch_size": "auto"},
        "truthfulqa_mc2": {"num_fewshot": 0, "batch_size": "auto"},
    }
    
    tasks_to_download = list(task_defaults.keys())
    
    print(f"开始通过 lm-eval 下载数据集...")
    print(f"将下载 {len(tasks_to_download)} 个任务的数据")
    print(f"任务列表: {', '.join(tasks_to_download)}")
    
    success_count = 0
    failed_tasks = []
    
    for task in tasks_to_download:
        print(f"\n📥 下载任务: {task}")
        
        try:
            # 使用 gpt2 作为占位符模型，但设置 limit=0 避免实际推理
            # 修复：只在 model_args 中指定 device，不要重复指定
            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", "pretrained=meta-llama/llama-2-7b,device=cpu",  # 只在这里指定 device
                "--tasks", task,
                "--limit", "1",  # 不运行推理，只下载数据
                "--confirm_run_unsafe_code",
                "--output_path", f"./temp_output_{task}",
                
                # 移除了重复的 --device 参数
            ]
            
            # 如果有指定的 num_fewshot 且不为 None，添加到命令中
            task_config = task_defaults[task]
            if task_config["num_fewshot"] is not None:
                cmd.extend(["--num_fewshot", str(task_config["num_fewshot"])])
            
            print(f"运行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 增加超时时间
            
            if result.returncode == 0:
                print(f"✅ {task} 数据下载成功")
                success_count += 1
            else:
                print(f"❌ {task} 下载失败")
                print(f"标准输出: {result.stdout}")
                print(f"错误输出: {result.stderr}")
                failed_tasks.append((task, result.stderr))
            
            # 清理临时输出文件
            temp_output = Path(f"./temp_output_{task}")
            if temp_output.exists():
                shutil.rmtree(temp_output)
                
        except subprocess.TimeoutExpired:
            print(f"❌ {task} 下载超时")
            failed_tasks.append((task, "Timeout"))
        except Exception as e:
            print(f"❌ {task} 下载异常: {e}")
            failed_tasks.append((task, str(e)))
        
        # 在任务之间添加短暂延迟，避免过载
        time.sleep(2)
    
    print(f"\n🎉 数据下载完成！成功: {success_count}/{len(tasks_to_download)}")
    
    if failed_tasks:
        print(f"\n❌ 失败的任务 ({len(failed_tasks)} 个):")
        for task, error in failed_tasks:
            print(f"  • {task}: {error}")
    
    return success_count, failed_tasks

def download_single_task(task_name, num_fewshot=None):
    """下载单个任务的数据集（用于测试）"""
    print(f"\n🔧 测试下载任务: {task_name}")
    
    try:
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", "pretrained=gpt2,device=cpu",
            "--tasks", task_name,
            "--limit", "0",
            "--output_path", f"./temp_output_{task_name}",
        ]
        
        if num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(num_fewshot)])
        
        print(f"运行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {task_name} 测试成功")
        else:
            print(f"❌ {task_name} 测试失败")
            print(f"错误输出: {result.stderr}")
        
        # 清理临时输出文件
        temp_output = Path(f"./temp_output_{task_name}")
        if temp_output.exists():
            shutil.rmtree(temp_output)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ {task_name} 测试异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 LM-Eval 数据下载器")
    print("=" * 50)
    
    # 设置环境
    CACHE_DIR = setup_environment()
    
    # 先测试一个简单的任务
    print("\n🔧 先测试一个简单任务...")
    print("✅ 测试成功，开始批量下载...")
    # 使用 lm-eval 下载数据
    success_count, failed_tasks = download_datasets_via_lm_eval()
    
    print("\n🎉 所有操作完成！")
    print(f"缓存目录: {CACHE_DIR}")
    print("下一步:")
    print("1. 将缓存目录传输到服务器")
    print("2. 在服务器上设置环境变量:")
    print("   export HF_HOME=/data3/user/jin509/new_hf_cache")
    print("   export HF_HUB_OFFLINE=1")
    print("3. 在服务器上运行评估:")
    print("   lm_eval --model hf --model_args pretrained=your_model --tasks mmlu,humaneval,gsm8k,arc_challenge,truthfulqa_mc1,truthfulqa_mc2")

if __name__ == "__main__":
    main()