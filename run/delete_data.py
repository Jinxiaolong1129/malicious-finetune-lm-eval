#!/usr/bin/env python3
import os
import shutil
import sys

def cleanup_directories(base_path="/home/jin509/llm_eval/lm-evaluation-harness/lm_eval/tasks"):
    """
    删除指定路径下除了保留列表中的所有一级目录
    
    Args:
        base_path: 目标路径，默认为 "lm_eval/tasks"
    """
    
    # 需要保留的目录列表
    keep_dirs = {
        "humaneval",
        "mmlu", 
        "truthfulqa",
        "arc",
        "gsm8k"
    }
    
    # 检查目标路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 路径 '{base_path}' 不存在")
        return False
    
    if not os.path.isdir(base_path):
        print(f"错误: '{base_path}' 不是一个目录")
        return False
    
    print(f"正在处理目录: {base_path}")
    print(f"将保留以下目录: {', '.join(sorted(keep_dirs))}")
    
    # 获取所有一级目录
    try:
        all_items = os.listdir(base_path)
    except PermissionError:
        print(f"错误: 没有权限访问目录 '{base_path}'")
        return False
    
    # 筛选出需要删除的目录
    dirs_to_delete = []
    files_found = []
    
    for item in all_items:
        item_path = os.path.join(base_path, item)
        
        if os.path.isdir(item_path):
            # 跳过特殊目录（如 __pycache__）和保留目录
            if item.startswith('__') and item.endswith('__'):
                print(f"跳过特殊目录: {item}")
                continue
            elif item in keep_dirs:
                print(f"保留目录: {item}")
                continue
            else:
                dirs_to_delete.append(item)
        else:
            files_found.append(item)
    
    # 显示将要删除的目录
    if dirs_to_delete:
        print(f"\n将删除以下 {len(dirs_to_delete)} 个目录:")
        for dir_name in sorted(dirs_to_delete):
            print(f"  - {dir_name}")
    else:
        print("\n没有需要删除的目录")
        return True
    
    # 显示发现的文件（不会删除）
    if files_found:
        print(f"\n发现以下文件（将保留）:")
        for file_name in sorted(files_found):
            print(f"  - {file_name}")
    
    # 确认删除
    print(f"\n准备删除 {len(dirs_to_delete)} 个目录，此操作不可逆！")
    confirm = input("确认删除？输入 'yes' 继续，其他任何输入取消: ").strip().lower()
    
    if confirm != 'yes':
        print("操作已取消")
        return False
    
    # 执行删除
    deleted_count = 0
    failed_count = 0
    
    for dir_name in dirs_to_delete:
        dir_path = os.path.join(base_path, dir_name)
        try:
            print(f"正在删除: {dir_name}")
            shutil.rmtree(dir_path)
            deleted_count += 1
        except PermissionError:
            print(f"错误: 没有权限删除 '{dir_name}'")
            failed_count += 1
        except Exception as e:
            print(f"错误: 删除 '{dir_name}' 时发生错误: {e}")
            failed_count += 1
    
    # 显示结果
    print(f"\n删除完成!")
    print(f"成功删除: {deleted_count} 个目录")
    if failed_count > 0:
        print(f"删除失败: {failed_count} 个目录")
    
    return failed_count == 0

def main():
    """主函数"""
    # 可以通过命令行参数指定路径
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "/home/jin509/llm_eval/lm-evaluation-harness/lm_eval/tasks"
    
    print("=" * 60)
    print("目录清理脚本")
    print("=" * 60)
    
    success = cleanup_directories(base_path)
    
    if success:
        print("\n所有操作完成！")
    else:
        print("\n操作过程中遇到错误，请检查上述信息")
        sys.exit(1)

if __name__ == "__main__":
    main()