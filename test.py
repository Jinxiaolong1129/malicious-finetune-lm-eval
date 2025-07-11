#!/usr/bin/env python3
"""
诊断和修复 HuggingFace 缓存问题的脚本
"""

import os
import sys
from pathlib import Path

# 设置环境变量
os.environ["HF_HOME"] = "/data3/user/jin509/new_hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data3/user/jin509/new_hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data3/user/jin509/new_hf_cache/hub"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def check_cache_structure():
    """检查缓存目录结构"""
    cache_dir = Path("/data3/user/jin509/new_hf_cache")
    
    print("=== 缓存目录结构检查 ===")
    
    # 检查主要目录
    for subdir in ["datasets", "hub", "modules"]:
        subpath = cache_dir / subdir
        if subpath.exists():
            print(f"✅ {subdir}/ 存在")
        else:
            print(f"❌ {subdir}/ 不存在")
    
    # 检查 humaneval 数据集
    humaneval_paths = [
        cache_dir / "datasets" / "openai_humaneval",
        cache_dir / "hub" / "datasets--openai_humaneval"
    ]
    
    print("\n=== HumanEval 数据集检查 ===")
    for path in humaneval_paths:
        if path.exists():
            print(f"✅ {path.relative_to(cache_dir)} 存在")
            # 列出内容
            try:
                items = list(path.iterdir())
                print(f"   内容: {[item.name for item in items[:5]]}")
            except Exception as e:
                print(f"   无法列出内容: {e}")
        else:
            print(f"❌ {path.relative_to(cache_dir)} 不存在")

def test_dataset_loading():
    """测试数据集加载"""
    print("\n=== 测试数据集加载 ===")
    
    try:
        from datasets import load_dataset
        
        # 尝试加载 humaneval
        print("尝试加载 openai_humaneval...")
        dataset = load_dataset("openai_humaneval", trust_remote_code=True)
        print(f"✅ 成功加载，splits: {list(dataset.keys())}")
        
        # 显示样本数量
        if 'test' in dataset:
            print(f"   test split 样本数: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print(f"   错误类型: {type(e).__name__}")

def fix_cache_permissions():
    """修复缓存目录权限"""
    print("\n=== 修复缓存目录权限 ===")
    
    cache_dir = Path("/data3/user/jin509/new_hf_cache")
    
    try:
        # 确保目录存在并具有正确权限
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 递归设置权限
        os.system(f"chmod -R 755 {cache_dir}")
        print("✅ 权限修复完成")
        
    except Exception as e:
        print(f"❌ 权限修复失败: {e}")

def main():
    """主函数"""
    print("HuggingFace 缓存诊断工具")
    print("=" * 50)
    
    # 显示环境变量
    print("=== 环境变量 ===")
    for key in ["HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE"]:
        print(f"{key}: {os.environ.get(key, 'NOT SET')}")
    
    # 检查缓存结构
    check_cache_structure()
    
    # 修复权限
    fix_cache_permissions()
    
    # 测试数据集加载
    test_dataset_loading()
    
    print("\n=== 建议的解决方案 ===")
    print("1. 如果数据集加载失败，请检查数据集是否正确下载")
    print("2. 确保所有环境变量都正确设置")
    print("3. 检查目录权限是否正确")
    print("4. 考虑清理并重新下载数据集")

if __name__ == "__main__":
    main()