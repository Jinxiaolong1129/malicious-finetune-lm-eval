#!/usr/bin/env python3
# ray-run_evaluation.py

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import argparse
import shutil
import json
import tempfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lm_eval import simple_evaluate
from lm_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
)
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import (
    get_file_datetime,
    get_file_task_name,
    get_results_filenames,
    get_sample_results_filenames,
    handle_non_serializable,
    hash_string,
    sanitize_list,
    sanitize_model_name,
    sanitize_task_name,
)

class LoRAEvaluator:
    """LoRA 模型评测器：合并-评测-清理一体化"""
    
    def __init__(self, base_model_name, lora_path):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.merged_path = None
        self.results = None
        self.log_samples = True  # 添加这个属性
        
    def merge_lora(self, temp_dir=None):
        """合并 LoRA 权重到临时目录"""
        if temp_dir is None:
            # 在 lora_path 下创建临时目录
            lora_parent_dir = Path(self.lora_path).parent
            temp_dir = tempfile.mkdtemp(prefix="merged_lora_", dir=str(lora_parent_dir))
        
        self.merged_path = temp_dir
        
        print(f"🔄 步骤1: 加载基础模型 {self.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"🔄 步骤2: 加载 LoRA 适配器 {self.lora_path}")
        model = PeftModel.from_pretrained(base_model, self.lora_path)
        
        print(f"🔄 步骤3: 合并 LoRA 权重")
        merged_model = model.merge_and_unload()
        
        print(f"🔄 步骤4: 保存到临时目录 {self.merged_path}")
        os.makedirs(self.merged_path, exist_ok=True)
        
        # 保存模型
        merged_model.save_pretrained(
            self.merged_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # 保存 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        tokenizer.save_pretrained(self.merged_path)
        
        # 清理内存
        del base_model, model, merged_model, tokenizer
        torch.cuda.empty_cache()
        
        print(f"✅ 模型合并完成，临时保存在: {self.merged_path}")
        return self.merged_path
    
    def evaluate(self, 
                 tasks=["mmlu"], 
                 tensor_parallel_size=1, 
                 gpu_memory_utilization=0.8,
                 num_fewshot=0,
                 **eval_kwargs):
        """使用 vLLM 评测合并后的模型"""
        if not self.merged_path or not os.path.exists(self.merged_path):
            raise ValueError("请先调用 merge_lora() 合并模型")
        
        print(f"\n🚀 步骤5: 使用 vLLM 开始评测...")
        print(f"📊 评测任务: {', '.join(tasks)}")
        print(f"⚡ Tensor Parallel Size: {tensor_parallel_size}")
        print(f"🧠 GPU Memory Utilization: {gpu_memory_utilization}")
        print(f"🎯 Few-shot: {num_fewshot}")
        
        # 构建 vLLM 模型参数
        model_args = {
            "pretrained": self.merged_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": 256,            
            "max_num_batched_tokens": 4096, 
        }
        
        # 构建评测参数
        eval_args = {
            "model": "vllm",
            "model_args": model_args,
            "tasks": tasks,
            "batch_size": "auto",
            "num_fewshot": num_fewshot,
            "log_samples": True
        }
        
        # 合并用户提供的参数
        eval_args.update(eval_kwargs)
        
        try:
            self.results = simple_evaluate(**eval_args)
            print("✅ 评测完成!")
            return self.results
        except Exception as e:
            print(f"❌ 评测过程中出错: {e}")
            raise
    
    def cleanup(self):
        """清理临时文件"""
        if self.merged_path and os.path.exists(self.merged_path):
            print(f"🧹 步骤6: 清理临时文件 {self.merged_path}")
            try:
                shutil.rmtree(self.merged_path)
                print("✅ 临时文件清理完成")
            except Exception as e:
                print(f"⚠️  清理临时文件时出错: {e}")
        else:
            print("ℹ️  没有需要清理的临时文件")
    
    def save_results(self, output_file):
        """保存评测结果到指定文件"""
        if self.results is None:
            print("⚠️  没有评测结果可保存")
            return
        
        try:
            print(f"💾 保存结果到: {output_file}")
            
            # 处理 samples 数据
            samples = None
            results_copy = self.results.copy()
            if self.log_samples and "samples" in results_copy:
                samples = results_copy.pop("samples")
            
            # 计算任务哈希值（如果有 samples）
            task_hashes = {}
            if samples:
                for task_name, task_samples in samples.items():
                    sample_hashes = [
                        s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                        for s in task_samples
                    ]
                    task_hashes[task_name] = hash_string("".join(sample_hashes))
            
            # 更新结果字典
            results_copy.update({"task_hashes": task_hashes})
            
            # 添加时间戳
            from datetime import datetime
            date_id = datetime.now().isoformat().replace(":", "-")
            results_copy.update({"evaluation_time": date_id})
            
            # 序列化结果
            dumped = json.dumps(
                results_copy, 
                indent=2, 
                default=handle_non_serializable, 
                ensure_ascii=False
            )
            
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 添加时间戳到文件名
            if output_path.suffix == ".json":
                final_output_file = output_path.with_name(f"{output_path.stem}_{date_id}.json")
            else:
                final_output_file = output_path.with_suffix(f"_{date_id}.json")
            
            # 写入文件
            with open(final_output_file, "w", encoding="utf-8") as f:
                f.write(dumped)
            
            print(f"✅ 结果已保存到: {final_output_file}")
            
            # 如果需要保存 samples 数据，单独保存
            if samples:
                samples_file = final_output_file.with_name(f"{final_output_file.stem}_samples.json")
                samples_dumped = json.dumps(
                    samples, 
                    indent=2, 
                    default=handle_non_serializable, 
                    ensure_ascii=False
                )
                with open(samples_file, "w", encoding="utf-8") as f:
                    f.write(samples_dumped)
                print(f"✅ 样本数据已保存到: {samples_file}")
            
            # 打印结果表格
            print("\n" + make_table(self.results))
            
        except Exception as e:
            print(f"❌ 保存结果时出错: {e}")
            import traceback
            traceback.print_exc()
            
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
            # 移除mmlu_前缀让显示更简洁
            display_name = task_name.replace("mmlu_", "")
            print(f"  {display_name:<35}: {acc:.4f}")
    
    def run_full_pipeline(self, tasks=["mmlu"], output_file=None, **eval_kwargs):
        """运行完整的评测流程：合并-评测-清理"""
        try:
            # 步骤1-4: 合并
            self.merge_lora()
            
            # 步骤5: 评测
            self.evaluate(tasks=tasks, **eval_kwargs)
            
            # 保存结果
            if output_file:
                self.save_results(output_file)
            
            # 打印摘要
            self.print_summary()
            
            return self.results
            
        finally:
            # 步骤6: 清理（无论是否出错都会执行）
            self.cleanup()

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="LoRA模型评测脚本")
    
    parser.add_argument("--base-model", type=str, required=True,
                        help="基础模型名称或路径")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="LoRA模型路径")
    parser.add_argument("--task", type=str, default="mmlu",
                        help="评测任务")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="GPU内存使用率")
    parser.add_argument("--num-fewshot", type=int, default=0,
                        help="Few-shot数量")
    parser.add_argument("--batch-size", type=str, default="auto",
                        help="批处理大小")
    
    return parser

def main():
    """主函数 - 支持命令行参数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 处理任务参数
    if args.task == "mmlu":
        tasks = ["mmlu"]
    elif args.task == "humaneval":
        tasks = ["humaneval"]
    elif args.task == "truthfulqa":
        tasks = ["truthfulqa_mc"]
    elif args.task == "all":
        tasks = ["mmlu", "humaneval", "truthfulqa_mc"]
    else:
        tasks = [args.task]  # 支持自定义任务
    
    print("🎯 开始 LoRA 模型自动化评测流程")
    print(f"📁 基础模型: {args.base_model}")
    print(f"📁 LoRA 路径: {args.lora_path}")
    print(f"📊 评测任务: {', '.join(tasks)}")
    print(f"⚡ Tensor Parallel: {args.tensor_parallel_size}")
    print(f"🧠 GPU内存使用率: {args.gpu_memory_utilization}")
    print(f"🎯 Few-shot数量: {args.num_fewshot}")
    print(f"💾 输出文件: {args.output}")
    print(f"{'='*60}")
    
    # 创建评测器
    evaluator = LoRAEvaluator(args.base_model, args.lora_path)
    
    # 构建评测参数
    eval_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "num_fewshot": args.num_fewshot
    }
    
    if args.batch_size != "auto":
        eval_kwargs["batch_size"] = args.batch_size
    
    # 运行完整流程
    try:
        results = evaluator.run_full_pipeline(
            tasks=tasks,
            output_file=args.output,
            **eval_kwargs
        )
        
        print(f"\n🎉 评测流程完成！")
        return results
        
    except Exception as e:
        print(f"❌ 评测流程失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()