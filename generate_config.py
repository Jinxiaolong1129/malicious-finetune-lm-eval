#!/usr/bin/env python3
"""
自动遍历模型训练结果目录，生成配置文件
使用方法: python generate_config.py [root_directory]
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def find_adapter_directories(root_dir: str) -> List[str]:
    """
    递归查找所有包含adapter_config.json的目录
    """
    adapter_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        if "adapter_config.json" in files and "adapter_model.safetensors" in files:
            adapter_dirs.append(root)
    
    return adapter_dirs

def extract_checkpoint_info(adapter_dir: str) -> tuple:
    """
    从路径中提取checkpoint信息，并确定epoch
    返回 (checkpoint_number, epoch, is_checkpoint)
    """
    path_parts = adapter_dir.split(os.sep)
    
    # 查找checkpoint目录
    checkpoint_part = None
    for part in path_parts:
        if part.startswith("checkpoint-"):
            checkpoint_part = part
            break
    
    if checkpoint_part:
        try:
            checkpoint_num = int(checkpoint_part.split("-")[1])
            # 查找同级的所有checkpoint目录来确定epoch
            parent_dir = os.path.dirname(adapter_dir)
            checkpoint_dirs = []
            
            if os.path.exists(parent_dir):
                for item in os.listdir(parent_dir):
                    if item.startswith("checkpoint-") and os.path.isdir(os.path.join(parent_dir, item)):
                        try:
                            num = int(item.split("-")[1])
                            checkpoint_dirs.append(num)
                        except ValueError:
                            continue
            
            # 排序checkpoint编号，找到当前checkpoint的位置
            checkpoint_dirs.sort()
            if checkpoint_num in checkpoint_dirs:
                epoch = checkpoint_dirs.index(checkpoint_num) + 1  # epoch从1开始
            else:
                epoch = 1
            
            return checkpoint_num, epoch, True
        except (ValueError, IndexError):
            pass
    
    # 如果不是checkpoint目录，可能是最终模型
    return None, None, False

def read_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    """
    读取adapter_config.json文件
    """
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {config_path}: {e}")
        return {}

def generate_experiment_name(lora_path: str, root_dir: str, checkpoint_num: int = None, epoch: int = None) -> str:
    """
    基于路径生成实验名称
    """
    # 移除根目录部分
    relative_path = os.path.relpath(lora_path, root_dir)
    parts = relative_path.split(os.sep)
    
    # 提取有意义的部分作为实验名称
    if len(parts) >= 1:
        experiment_type = parts[0]
        
        # 如果有更多层级，可以组合起来
        if len(parts) > 3:
            # 例如: bea, booster/pre_alignment_model, etc.
            if "pre_alignment_model" in parts:
                experiment_type = f"{experiment_type}_pre_alignment"
            elif len(parts) > 4:
                # 检查是否有特殊的配置目录名
                config_part = parts[-1] if not parts[-1].startswith("checkpoint-") else parts[-2]
                if config_part not in ["bea", experiment_type]:
                    experiment_type = f"{experiment_type}_{config_part}"
        
        # 如果是checkpoint，添加epoch信息
        if checkpoint_num is not None and epoch is not None:
            experiment_type = f"{experiment_type}_epoch{epoch}"
        elif checkpoint_num is None and epoch is None:
            # 可能是final model
            experiment_type = f"{experiment_type}_final"
        
        return experiment_type
    
    return "unknown"

def generate_model_configs(root_dir: str = "experiments-back") -> List[Dict[str, str]]:
    """
    生成模型配置列表
    """
    print(f"Scanning directory: {root_dir}")
    
    adapter_dirs = find_adapter_directories(root_dir)
    print(f"Found {len(adapter_dirs)} adapter directories")
    
    configs = []
    
    for adapter_dir in adapter_dirs:
        print(f"Processing: {adapter_dir}")
        
        # 读取adapter配置
        adapter_config = read_adapter_config(adapter_dir)
        
        # 获取base model信息
        base_model = adapter_config.get("base_model_name_or_path", "Unknown")
        
        # 提取checkpoint和epoch信息
        checkpoint_num, epoch, is_checkpoint = extract_checkpoint_info(adapter_dir)
        
        # 生成实验名称
        experiment_name = generate_experiment_name(adapter_dir, root_dir, checkpoint_num, epoch)
        
        # 标准化路径
        lora_path = adapter_dir.replace("\\", "/")
        
        config = {
            "experiment_name": experiment_name,
            "base_model": base_model,
            "lora_path": lora_path
        }
        
        # 添加epoch和checkpoint信息
        if is_checkpoint:
            config["checkpoint"] = checkpoint_num
            config["epoch"] = epoch
        else:
            config["checkpoint"] = "final"
            config["epoch"] = "final"
        
        configs.append(config)
    
    # 按实验名称和epoch排序
    configs.sort(key=lambda x: (x["experiment_name"], x["epoch"] if isinstance(x["epoch"], int) else 999))
    
    return configs

def save_configs(configs: List[Dict[str, str]], output_file: str = "model_configs.json"):
    """
    保存配置到JSON文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration saved to: {output_file}")

def print_summary(configs: List[Dict[str, str]]):
    """
    打印配置摘要
    """
    print(f"\n=== Configuration Summary ===")
    print(f"Total configurations: {len(configs)}")
    
    # 按base model分组统计
    base_model_counts = {}
    for config in configs:
        base_model = config["base_model"]
        base_model_counts[base_model] = base_model_counts.get(base_model, 0) + 1
    
    print(f"\nBase models:")
    for model, count in base_model_counts.items():
        print(f"  {model}: {count} configurations")
    
    # 统计checkpoint vs final models
    checkpoint_count = sum(1 for config in configs if config["checkpoint"] != "final")
    final_count = sum(1 for config in configs if config["checkpoint"] == "final")
    
    print(f"\nModel types:")
    print(f"  Checkpoint models: {checkpoint_count}")
    print(f"  Final models: {final_count}")
    
    print(f"\nExperiment types:")
    experiment_types = set()
    for config in configs:
        # 移除epoch后缀来获取基础实验类型
        exp_name = config["experiment_name"]
        if "_epoch" in exp_name:
            base_exp = exp_name.split("_epoch")[0]
        else:
            base_exp = exp_name.replace("_final", "")
        experiment_types.add(base_exp)
    
    for exp_type in sorted(experiment_types):
        # 统计该实验类型的配置数量
        count = sum(1 for config in configs 
                   if config["experiment_name"].startswith(exp_type))
        print(f"  {exp_type}: {count} configurations")

def main():
    parser = argparse.ArgumentParser(description="Generate model configuration file from training results")
    parser.add_argument("root_dir", nargs="?", default="/data3/user/jin509/malicious-finetuning/experiments", 
                       help="Root directory containing training results (default: experiments-back)")
    parser.add_argument("-o", "--output", default="model_configs-new.json",
                       help="Output configuration file (default: model_configs.json)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only print configurations without saving")
    parser.add_argument("--final-only", action="store_true",
                       help="Only include final models (not checkpoints)")
    parser.add_argument("--checkpoints-only", action="store_true",
                       help="Only include checkpoint models (not final)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        print(f"Error: Directory '{args.root_dir}' does not exist")
        return 1
    
    # 生成配置
    configs = generate_model_configs(args.root_dir)
    
    if not configs:
        print("No adapter configurations found!")
        return 1
    
    # 过滤配置
    if args.final_only:
        configs = [c for c in configs if c["checkpoint"] == "final"]
        print(f"Filtered to final models only: {len(configs)} configurations")
    elif args.checkpoints_only:
        configs = [c for c in configs if c["checkpoint"] != "final"]
        print(f"Filtered to checkpoint models only: {len(configs)} configurations")
    
    # 打印摘要
    print_summary(configs)
    
    # 保存配置（如果不是dry run）
    if not args.dry_run:
        save_configs(configs, args.output)
    else:
        print(f"\n=== Dry Run - Configuration Content ===")
        print(json.dumps(configs, indent=2, ensure_ascii=False))
    
    return 0

if __name__ == "__main__":
    exit(main())