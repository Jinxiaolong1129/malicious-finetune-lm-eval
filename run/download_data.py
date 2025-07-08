#!/usr/bin/env python3
"""
简单的数据集下载脚本
专门用于下载 lm-eval 所需的数据集
"""

import os
# TODO set HF_HOME here
os.environ["HF_HOME"] = "/data3/user/jin509/new_hf_cache"

from pathlib import Path
from datasets import load_dataset


def download_all_datasets():
    """下载所有必需的数据集"""
    
    # 确保缓存目录存在
    cache_dir = "/data3/user/jin509/new_hf_cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ HF 缓存目录: {cache_dir}")
    
    datasets_to_download = [
        ("mmlu", "cais/mmlu", "all"),  # 修复：使用 "all" 配置
        ("humaneval", "openai_humaneval", None),
        ("gsm8k", "gsm8k", "main"),
        ("arc_challenge", "ai2_arc", "ARC-Challenge"),
        ("truthfulqa", "truthful_qa", "multiple_choice"),
    ]
    
    print("开始下载数据集...\n")
    
    success_count = 0
    for name, dataset_id, config in datasets_to_download:
        try:
            print(f"📥 下载 {name}...")
            
            if config:
                dataset = load_dataset(dataset_id, config)
            else:
                dataset = load_dataset(dataset_id)
            
            # 显示数据集信息
            if hasattr(dataset, 'keys'):
                splits = list(dataset.keys())
                total_examples = sum(len(dataset[split]) for split in splits)
                print(f"✅ {name} 完成 (splits: {splits}, 总样本数: {total_examples})")
            else:
                print(f"✅ {name} 完成")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
    
    print(f"\n🎉 数据集下载完成！成功: {success_count}/{len(datasets_to_download)}")
    
    # 检查缓存目录
    print(f"\n缓存目录内容:")
    try:
        cache_path = Path(cache_dir)
        for item in sorted(cache_path.glob("**/*")):
            if item.is_dir() and not item.name.startswith('.'):
                print(f"  📁 {item.relative_to(cache_path)}")
    except Exception as e:
        print(f"  无法列出缓存目录: {e}")

if __name__ == "__main__":
    download_all_datasets()