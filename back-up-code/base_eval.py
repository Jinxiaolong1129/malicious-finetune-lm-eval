#!/usr/bin/env python3

import os
import json
import multiprocessing
from pathlib import Path

# 设置环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置启动方法
multiprocessing.set_start_method('spawn', force=True)

from lm_eval import simple_evaluate

class BaseModelEvaluator:
    """基础模型评测器：使用 vLLM 直接评测预训练模型"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.results = None
        
    def evaluate(self, tasks=["mmlu"], **eval_kwargs):
        """使用 vLLM 评测基础模型"""
        print(f"\n🚀 开始使用 vLLM 评测模型...")
        print(f"🤖 模型: {self.model_name}")
        print(f"📊 评测任务: {', '.join(tasks)}")
        
        # 默认的 vLLM 参数
        default_model_args = {
            "pretrained": self.model_name,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 256,            
            "max_num_batched_tokens": 4096, 
        }
        
        # 合并用户提供的模型参数
        model_args = default_model_args.copy()
        if "model_args" in eval_kwargs:
            model_args.update(eval_kwargs.pop("model_args"))
        
        # 默认的评测参数
        default_eval_args = {
            "model": "vllm",
            "model_args": model_args,
            "tasks": tasks,
            "batch_size": "auto",
            "num_fewshot": 5,  # MMLU 标准使用 5-shot
            "log_samples": True
        }
        
        # 合并用户提供的评测参数
        final_eval_args = default_eval_args.copy()
        final_eval_args.update(eval_kwargs)
        
        try:
            self.results = simple_evaluate(**final_eval_args)
            print("✅ 评测完成!")
            return self.results
        except Exception as e:
            print(f"❌ 评测过程中出错: {e}")
            raise
    
    def save_results(self, output_file):
        """保存评测结果"""
        if self.results:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"💾 结果已保存到: {output_file}")
        else:
            print("⚠️  没有评测结果可保存")
    
    def print_summary(self):
        """打印评测结果摘要"""
        if not self.results:
            print("⚠️  没有评测结果")
            return
        
        print(f"\n{'='*60}")
        print("📊 评测结果摘要")
        print(f"{'='*60}")
        
        # 收集所有准确率
        accuracies = []
        task_results = []
        
        for task_name, task_results_dict in self.results.get("results", {}).items():
            acc = task_results_dict.get("acc", 0.0)
            accuracies.append(acc)
            task_results.append((task_name, acc))
            
        # 按准确率排序
        task_results.sort(key=lambda x: x[1], reverse=True)
        
        # 计算总体统计
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            print(f"\n🎯 总体表现:")
            print(f"  平均准确率: {avg_acc:.4f}")
            print(f"  最高准确率: {max(accuracies):.4f}")
            print(f"  最低准确率: {min(accuracies):.4f}")
            print(f"  评测任务数: {len(accuracies)}")
        
        # 显示各任务详细结果
        print(f"\n📋 各任务详细结果 (按准确率排序):")
        print("-" * 60)
        for task_name, acc in task_results:
            # 移除任务前缀让显示更简洁
            display_name = task_name.replace("mmlu_", "").replace("hellaswag_", "")
            print(f"  {display_name:<35}: {acc:.4f}")
    
    def run_evaluation(self, tasks=["mmlu"], output_file=None, **eval_kwargs):
        """运行完整的评测流程"""
        try:
            # 评测
            self.evaluate(tasks=tasks, **eval_kwargs)
            
            # 保存结果
            if output_file:
                self.save_results(output_file)
            
            # 打印摘要
            self.print_summary()
            
            return self.results
            
        except Exception as e:
            print(f"❌ 评测流程失败: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    
    # 配置参数 - 修改为您要评测的模型
    model_name = "meta-llama/Llama-3.1-8B"
    
    # 评测任务 - 可以选择多个任务
    tasks = ["mmlu"]  # 可选: ["mmlu", "hellaswag", "arc", "winogrande"]
    
    # 输出文件
    output_file = "base_model_evaluation_results.json"
    
    print("🎯 开始基础模型评测流程")
    print(f"🤖 模型: {model_name}")
    print(f"📊 评测任务: {', '.join(tasks)}")
    print(f"🎯 评测模式: 5-shot (MMLU标准)")
    print(f"💾 输出文件: {output_file}")
    print(f"{'='*60}")
    
    # 创建评测器
    evaluator = BaseModelEvaluator(model_name)
    
    # 运行评测
    results = evaluator.run_evaluation(
        tasks=tasks,
        output_file=output_file,
        # 可以在这里添加自定义参数，例如:
        # num_fewshot=0,  # 改为 0-shot 评测
        # model_args={"tensor_parallel_size": 2}  # 调整并行度
    )
    
    if results:
        print(f"\n🎉 评测流程完成！")
    
    return results

if __name__ == "__main__":
    main()