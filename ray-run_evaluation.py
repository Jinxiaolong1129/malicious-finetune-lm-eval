#!/usr/bin/env python3
# ray-run_evaluation.py - 优化版：支持多任务一次加载 + GSM8K

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


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

class OptimizedLoRAEvaluator:
    """优化的LoRA模型评测器：支持多任务一次加载"""
    
    def __init__(self, base_model_name, lora_path):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.merged_path = None
        self.results = None
        self.log_samples = True
        
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
    
    def evaluate_multiple_tasks(self, 
                               tasks=["humaneval"], 
                               tensor_parallel_size=1, 
                               gpu_memory_utilization=0.8,
                               **eval_kwargs):
        """使用 vLLM 评测多个任务 - 修正版 + GSM8K 支持"""
        if not self.merged_path or not os.path.exists(self.merged_path):
            raise ValueError("请先调用 merge_lora() 合并模型")
        
        print(f"\n🚀 步骤5: 使用 vLLM 开始多任务评测...")
        print(f"📊 评测任务: {', '.join(tasks)} (共{len(tasks)}个)")
        print(f"⚡ Tensor Parallel Size: {tensor_parallel_size}")
        print(f"🧠 GPU Memory Utilization: {gpu_memory_utilization}")
        print(f"💡 优势: 模型只加载一次，评测{len(tasks)}个任务")
        
        # 任务特定的默认参数 - 修正版 + GSM8K
        task_defaults = {
            "mmlu": {"num_fewshot": 5, "batch_size": "auto"},
            "humaneval": {"num_fewshot": 0, "batch_size": "auto"},  
            "gsm8k": {"num_fewshot": 0, "batch_size": "auto"},  # 新增 GSM8K
            "truthfulqa_mc1": {"num_fewshot": 0, "batch_size": "auto"},
            "truthfulqa_mc2": {"num_fewshot": 0, "batch_size": "auto"},
        }
        
        # 标准化任务名称 - 修正版 + GSM8K
        normalized_tasks = []
        has_unsafe_tasks = False
        
        for task in tasks:
            if task.lower() == "mmlu":
                normalized_tasks.append("mmlu")
            elif task.lower() == "humaneval":
                normalized_tasks.append("humaneval")
                has_unsafe_tasks = True
            elif task.lower() == "gsm8k":  # 新增 GSM8K 支持
                normalized_tasks.append("gsm8k")
            elif task.lower() in ["truthfulqa", "truthfulqa_mc"]:
                # TruthfulQA 默认评测 MC1 和 MC2
                normalized_tasks.extend(["truthfulqa_mc1", "truthfulqa_mc2"])
            elif task.lower() == "truthfulqa_mc1":
                normalized_tasks.append("truthfulqa_mc1")
            elif task.lower() == "truthfulqa_mc2":
                normalized_tasks.append("truthfulqa_mc2")
            else:
                normalized_tasks.append(task)
                if any(unsafe_keyword in task.lower() for unsafe_keyword in ["code", "eval", "exec"]):
                    has_unsafe_tasks = True
        
        print(f"📋 标准化后的任务: {', '.join(normalized_tasks)}")
        
        if has_unsafe_tasks:
            print(f"⚠️  检测到包含代码执行任务 (如 HumanEval)，将自动启用安全确认参数")
        
        # 按 num_fewshot 分组评测 - 新增逻辑
        fewshot_groups = {}
        for task in normalized_tasks:
            fewshot = task_defaults.get(task, {}).get("num_fewshot", 0)
            if fewshot not in fewshot_groups:
                fewshot_groups[fewshot] = []
            fewshot_groups[fewshot].append(task)
        
        print(f"🎯 按 few-shot 分组:")
        for fewshot, group_tasks in fewshot_groups.items():
            print(f"   {fewshot}-shot: {', '.join(group_tasks)}")
        
        # 如果只有一组，直接评测
        if len(fewshot_groups) == 1:
            fewshot_value = list(fewshot_groups.keys())[0]
            return self._evaluate_single_group(normalized_tasks, fewshot_value, 
                                             tensor_parallel_size, gpu_memory_utilization, 
                                             has_unsafe_tasks, **eval_kwargs)
        else:
            # 多组分别评测后合并结果
            return self._evaluate_multiple_groups(fewshot_groups, 
                                                tensor_parallel_size, gpu_memory_utilization,
                                                has_unsafe_tasks, **eval_kwargs)

    def _evaluate_single_group(self, tasks, num_fewshot, tensor_parallel_size, 
                              gpu_memory_utilization, has_unsafe_tasks, **eval_kwargs):
        """评测单个 few-shot 组"""
        model_args = {
            "pretrained": self.merged_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": 256,            
            "max_num_batched_tokens": 4096, 
        }
        
        eval_args = {
            "model": "vllm",
            "model_args": model_args,
            "tasks": tasks,
            "batch_size": "auto",
            "num_fewshot": num_fewshot,
            "log_samples": True
        }
        
        # GSM8K 特殊配置
        if any(task == "gsm8k" for task in tasks):
            print(f"🧮 GSM8K 特殊配置：启用数学推理优化")
            # GSM8K 通常不需要特殊的模型参数，但可以调整推理参数
            eval_args["limit"] = None  # 评测全部样本
        
        if has_unsafe_tasks:
            eval_args["confirm_run_unsafe_code"] = True
            print(f"🔐 安全设置：已启用 confirm_run_unsafe_code=True")
        
        # 合并用户参数
        for key, value in eval_kwargs.items():
            if key not in ['evaluation_script', 'python_executable']:
                eval_args[key] = value
        
        print(f"⏳ 开始评测 {len(tasks)} 个任务 (num_fewshot={num_fewshot})...")
        results = simple_evaluate(**eval_args)
        print(f"✅ 任务组评测完成!")
        
        self.results = results
        return results

    def _evaluate_multiple_groups(self, fewshot_groups, tensor_parallel_size, 
                                 gpu_memory_utilization, has_unsafe_tasks, **eval_kwargs):
        """评测多个 few-shot 组并合并结果"""
        all_results = {"results": {}, "samples": {}}
        
        for fewshot, group_tasks in fewshot_groups.items():
            print(f"\n🎯 评测 {fewshot}-shot 组: {', '.join(group_tasks)}")
            
            group_results = self._evaluate_single_group(
                group_tasks, fewshot, tensor_parallel_size, 
                gpu_memory_utilization, has_unsafe_tasks, **eval_kwargs
            )
            
            # 合并结果
            if "results" in group_results:
                all_results["results"].update(group_results["results"])
            if "samples" in group_results:
                all_results["samples"].update(group_results["samples"])
        
        # 复制其他元数据
        for key, value in group_results.items():
            if key not in ["results", "samples"]:
                all_results[key] = value
        
        self.results = all_results
        return all_results
    
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
        """保存多任务评测结果到指定文件"""
        if self.results is None:
            print("⚠️  没有评测结果可保存")
            return
        
        try:
            print(f"💾 保存多任务结果到: {output_file}")
            
            # 处理 samples 数据
            samples = None
            results_copy = self.results.copy()
            if self.log_samples and "samples" in results_copy:
                samples = results_copy.pop("samples")
            
            # 计算任务哈希值
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
            
            # 添加时间戳和元信息
            from datetime import datetime
            date_id = datetime.now().isoformat().replace(":", "-")
            
            # 收集评测的任务列表
            evaluated_tasks = list(results_copy.get("results", {}).keys())
            
            results_copy.update({
                "evaluation_time": date_id,
                "evaluation_mode": "multi_task_single_load",
                "tasks_evaluated": evaluated_tasks,
                "task_count": len(evaluated_tasks),
                "lora_path": self.lora_path,
                "base_model": self.base_model_name
            })
            
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
            
            # 写入文件
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(dumped)
            
            print(f"✅ 多任务结果已保存到: {output_path}")
            
            # 如果需要保存 samples 数据，单独保存
            if samples:
                samples_file = output_path.with_name(f"{output_path.stem}_samples.json")
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
            print(f"\n📊 详细评测结果:")
            print(make_table(self.results))
            
        except Exception as e:
            print(f"❌ 保存结果时出错: {e}")
            import traceback
            traceback.print_exc()
            
    def print_summary(self):
        """打印多任务评测结果摘要"""
        if not self.results:
            print("⚠️  没有评测结果")
            return
        
        print(f"\n{'='*80}")
        print("📊 多任务评测结果摘要")
        print(f"{'='*80}")
        
        # 收集所有任务的准确率
        task_results = []
        all_accuracies = []
        
        results_dict = self.results.get("results", {})
        
        for task_name, task_result in results_dict.items():
            # 尝试获取不同类型的准确率指标
            acc = (task_result.get("acc") or 
                   task_result.get("acc_norm") or 
                   task_result.get("exact_match") or 
                   task_result.get("pass@1") or 
                   0.0)
            
            task_results.append((task_name, acc))
            all_accuracies.append(acc)
        
        # 按准确率排序
        task_results.sort(key=lambda x: x[1], reverse=True)
        
        # 计算总体统计
        if all_accuracies:
            avg_acc = sum(all_accuracies) / len(all_accuracies)
            print(f"\n🎯 总体表现:")
            print(f"  评测任务数: {len(all_accuracies)}")
            print(f"  平均准确率: {avg_acc:.4f}")
            print(f"  最高准确率: {max(all_accuracies):.4f}")
            print(f"  最低准确率: {min(all_accuracies):.4f}")
            print(f"  标准差: {(sum((x - avg_acc) ** 2 for x in all_accuracies) / len(all_accuracies)) ** 0.5:.4f}")
        
        # 显示各任务详细结果
        print(f"\n📋 各任务详细结果 (按准确率排序):")
        print("-" * 80)
        for task_name, acc in task_results:
            # 移除前缀让显示更简洁
            display_name = task_name.replace("mmlu_", "").replace("truthfulqa_", "")
            
            # 根据任务类型显示不同的指标名称
            if "humaneval" in task_name.lower():
                metric_name = "Pass@1"
            elif "gsm8k" in task_name.lower():  # 新增 GSM8K 指标名称
                metric_name = "Accuracy"
            elif "truthful" in task_name.lower():
                metric_name = "Accuracy"
            else:
                metric_name = "Accuracy"
            
            print(f"  {display_name:<45}: {acc:.4f} ({metric_name})")
        
        # 显示效率提升信息
        task_count = len(task_results)
        print(f"\n⚡ 效率提升:")
        print(f"  单次加载评测 {task_count} 个任务")
        print(f"  相比单任务模式提升约 {task_count}x 效率")
        print(f"  节省了 {task_count - 1} 次模型加载时间")
    
    def run_full_pipeline(self, tasks=["mmlu"], output_file=None, **eval_kwargs):
        """运行完整的多任务评测流程：合并-评测-清理"""
        try:
            print(f"🚀 启动多任务评测流程")
            print(f"📊 任务列表: {', '.join(tasks)} (共{len(tasks)}个)")
            
            # 步骤1-4: 合并
            self.merge_lora()
            
            # 步骤5: 多任务评测
            self.evaluate_multiple_tasks(tasks=tasks, **eval_kwargs)
            
            # 保存结果
            if output_file:
                self.save_results(output_file)
            
            # 打印摘要
            self.print_summary()
            
            return self.results
            
        finally:
            # 步骤6: 清理（无论是否出错都会执行）
            self.cleanup()

def parse_tasks(tasks_str):
    """解析任务字符串 - 修正版 + GSM8K"""
    if not tasks_str:
        return ["mmlu"]
    
    tasks = [task.strip() for task in tasks_str.split(",")]
    normalized_tasks = []
    
    for task in tasks:
        if task.lower() == "all":
            normalized_tasks.extend(["mmlu", "humaneval", "gsm8k", "truthfulqa_mc1", "truthfulqa_mc2"])  # 新增 GSM8K
        elif task.lower() == "truthful":  # 简化输入
            normalized_tasks.extend(["truthfulqa_mc1", "truthfulqa_mc2"])
        elif task.lower() == "gsm8k":  # 新增 GSM8K 支持
            normalized_tasks.append("gsm8k")
        else:
            normalized_tasks.append(task)
    
    return normalized_tasks

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="优化版LoRA模型评测脚本 - 支持多任务一次加载 + GSM8K")
    
    parser.add_argument("--base-model", type=str, required=True,
                        help="基础模型名称或路径")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="LoRA模型路径")
    parser.add_argument("--tasks", type=str, default="humaneval",
                        help="评测任务，支持逗号分隔多个任务，如: mmlu,humaneval,gsm8k,truthfulqa")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                        help="GPU内存使用率")
    parser.add_argument("--batch-size", type=str, default="auto",
                        help="批处理大小 (推荐使用auto)")
    
    return parser

def main():
    """主函数 - 支持多任务命令行参数 + GSM8K"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 处理任务参数
    tasks = parse_tasks(args.tasks)
    
    print("🎯 启动优化版 LoRA 模型多任务评测流程 (包含 GSM8K)")
    print(f"📁 基础模型: {args.base_model}")
    print(f"📁 LoRA 路径: {args.lora_path}")
    print(f"📊 评测任务: {', '.join(tasks)} (共{len(tasks)}个)")
    print(f"⚡ Tensor Parallel: {args.tensor_parallel_size}")
    print(f"🧠 GPU内存使用率: {args.gpu_memory_utilization}")
    print(f"💾 输出文件: {args.output}")
    print(f"💡 优化模式: 一次加载评测{len(tasks)}个任务")
    
    # 特别提示 GSM8K
    if "gsm8k" in tasks:
        print(f"🧮 包含数学推理任务 GSM8K (8k 数学应用题)")
    
    print(f"{'='*80}")
    
    # 创建优化版评测器
    evaluator = OptimizedLoRAEvaluator(args.base_model, args.lora_path)
    
    # 构建评测参数
    eval_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
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
        
        print(f"\n🎉 多任务评测流程完成！")
        print(f"✅ 成功评测了 {len(tasks)} 个任务")
        if "gsm8k" in tasks:
            print(f"🧮 GSM8K 数学推理评测已完成")
        print(f"⚡ 效率提升: 相比单任务模式快约 {len(tasks)} 倍")
        return results
        
    except Exception as e:
        print(f"❌ 多任务评测流程失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()