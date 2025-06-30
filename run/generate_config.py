#!/usr/bin/env python3
"""
自动遍历模型训练结果目录，生成配置文件
使用方法: python generate_config.py [root_directory]
支持分类保存：final模型、checkpoint模型、pre_alignment模型
输出文件保存在root_dir目录下，文件名格式为：lm_eval_model_configs_{type}.json
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

def classify_model_type(config: Dict[str, Any]) -> str:
    """
    根据配置信息对模型进行分类
    返回: 'final', 'checkpoint', 'pre_alignment'
    """
    experiment_name = config.get("experiment_name", "")
    lora_path = config.get("lora_path", "")
    
    # 检查是否为pre_alignment模型
    if "pre_alignment" in experiment_name.lower() or "pre_alignment" in lora_path.lower():
        return "pre_alignment"
    
    # 检查是否为checkpoint模型
    if config.get("checkpoint") != "final":
        return "checkpoint"
    
    # 默认为final模型
    return "final"

def generate_model_configs(root_dir: str = "experiments-back") -> Dict[str, List[Dict[str, Any]]]:
    """
    生成模型配置列表，按类型分组
    返回: {'final': [...], 'checkpoint': [...], 'pre_alignment': [...]}
    """
    print(f"Scanning directory: {root_dir}")
    
    adapter_dirs = find_adapter_directories(root_dir)
    print(f"Found {len(adapter_dirs)} adapter directories")
    
    # 按类型分组的配置
    classified_configs = {
        'final': [],
        'checkpoint': [],
        'pre_alignment': []
    }
    
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
        
        # 分类并添加到对应组
        model_type = classify_model_type(config)
        classified_configs[model_type].append(config)
    
    # 对每个分组进行排序
    for model_type in classified_configs:
        classified_configs[model_type].sort(
            key=lambda x: (x["experiment_name"], x["epoch"] if isinstance(x["epoch"], int) else 999)
        )
    
    return classified_configs

def save_classified_configs(classified_configs: Dict[str, List[Dict[str, Any]]], 
                          root_dir: str, output_prefix: str = "lm_eval_model_configs"):
    """
    分别保存不同类型的配置到root_dir目录下的不同文件
    """
    saved_files = []
    
    for model_type, configs in classified_configs.items():
        if configs:  # 只保存非空的配置
            filename = f"{output_prefix}_{model_type}.json"
            # 保存到root_dir目录下
            output_path = os.path.join(root_dir, filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=2, ensure_ascii=False)
            saved_files.append(output_path)
            print(f"Saved {len(configs)} {model_type} configurations to: {output_path}")
    
    return saved_files

def save_all_configs(classified_configs: Dict[str, List[Dict[str, Any]]], 
                    root_dir: str, output_file: str = "lm_eval_model_configs_all.json"):
    """
    保存所有配置到root_dir目录下的单个文件（保持原有功能）
    """
    all_configs = []
    for model_type, configs in classified_configs.items():
        all_configs.extend(configs)
    
    # 重新排序所有配置
    all_configs.sort(key=lambda x: (x["experiment_name"], x["epoch"] if isinstance(x["epoch"], int) else 999))
    
    # 保存到root_dir目录下
    output_path = os.path.join(root_dir, output_file)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)
    
    print(f"All configurations saved to: {output_path}")
    return output_path

def print_classified_summary(classified_configs: Dict[str, List[Dict[str, Any]]]):
    """
    打印分类配置摘要
    """
    total_configs = sum(len(configs) for configs in classified_configs.values())
    
    print(f"\n=== Classified Configuration Summary ===")
    print(f"Total configurations: {total_configs}")
    
    print(f"\nBy model type:")
    for model_type, configs in classified_configs.items():
        print(f"  {model_type.capitalize()} models: {len(configs)} configurations")
        
        if configs:
            # 统计该类型下的base model分布
            base_model_counts = {}
            for config in configs:
                base_model = config["base_model"]
                base_model_counts[base_model] = base_model_counts.get(base_model, 0) + 1
            
            if len(base_model_counts) > 1:
                print(f"    Base models:")
                for model, count in base_model_counts.items():
                    print(f"      {model}: {count}")
    
    # 统计实验类型
    print(f"\nExperiment types by category:")
    for model_type, configs in classified_configs.items():
        if configs:
            experiment_types = set()
            for config in configs:
                exp_name = config["experiment_name"]
                # 移除epoch和final后缀来获取基础实验类型
                if "_epoch" in exp_name:
                    base_exp = exp_name.split("_epoch")[0]
                else:
                    base_exp = exp_name.replace("_final", "")
                experiment_types.add(base_exp)
            
            print(f"  {model_type.capitalize()}:")
            for exp_type in sorted(experiment_types):
                count = sum(1 for config in configs 
                           if config["experiment_name"].startswith(exp_type))
                print(f"    {exp_type}: {count} configurations")

def main():
    parser = argparse.ArgumentParser(description="Generate model configuration files from training results")
    parser.add_argument("--root_dir", nargs="?", default="/data3/user/jin509/malicious-finetuning/experiments", 
                       help="Root directory containing training results")
    parser.add_argument("-o", "--output-prefix", default="lm_eval_model_configs",
                       help="Output file prefix (default: lm_eval_model_configs)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only print configurations without saving")
    parser.add_argument("--save-all", action="store_true",
                       help="Also save all configurations to a single file")
    parser.add_argument("--separate-only", action="store_true",
                       help="Only save separate files (default behavior)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        print(f"Error: Directory '{args.root_dir}' does not exist")
        return 1
    
    # 生成分类配置
    classified_configs = generate_model_configs(args.root_dir)
    
    # 检查是否有配置
    total_configs = sum(len(configs) for configs in classified_configs.values())
    if total_configs == 0:
        print("No adapter configurations found!")
        return 1
    
    # 打印摘要
    print_classified_summary(classified_configs)
    
    # 保存配置（如果不是dry run）
    if not args.dry_run:
        # 分别保存不同类型的配置到root_dir目录下
        saved_files = save_classified_configs(classified_configs, args.root_dir, args.output_prefix)
        
        # 如果需要，也保存所有配置到root_dir目录下的单个文件
        if args.save_all:
            all_file = save_all_configs(classified_configs, args.root_dir, f"{args.output_prefix}_all.json")
            saved_files.append(all_file)
        
        print(f"\nSaved files: {', '.join(saved_files)}")
        print(f"All files saved in directory: {args.root_dir}")
    else:
        print(f"\n=== Dry Run - Configuration Content ===")
        for model_type, configs in classified_configs.items():
            if configs:
                print(f"\n--- {model_type.upper()} MODELS ({len(configs)} configurations) ---")
                print(json.dumps(configs, indent=2, ensure_ascii=False))
    
    return 0

if __name__ == "__main__":
    exit(main())