#!/usr/bin/env python3
# ray-run_evaluation.py 

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
        """使用 vLLM 评测多个任务 - 修正版 + GSM8K + ARC Challenge 支持"""
        if not self.merged_path or not os.path.exists(self.merged_path):
            raise ValueError("请先调用 merge_lora() 合并模型")
        
        print(f"\n🚀 步骤5: 使用 vLLM 开始多任务评测...")
        print(f"📊 评测任务: {', '.join(tasks)} (共{len(tasks)}个)")
        print(f"⚡ Tensor Parallel Size: {tensor_parallel_size}")
        print(f"🧠 GPU Memory Utilization: {gpu_memory_utilization}")
        print(f"💡 优势: 模型只加载一次，评测{len(tasks)}个任务")
        
        # 任务特定的默认参数 
        task_defaults = {
            "mmlu": {"num_fewshot": 0, "batch_size": "auto"},
            "humaneval": {"num_fewshot": 0, "batch_size": "auto"},  
            "gsm8k": {"num_fewshot": 0, "batch_size": "auto"},
            "arc_challenge": {"num_fewshot": None, "batch_size": "auto"},  # 使用默认值
            "truthfulqa_mc1": {"num_fewshot": 0, "batch_size": "auto"},
            "truthfulqa_mc2": {"num_fewshot": 0, "batch_size": "auto"},
        }
        
        # 标准化任务名称
        normalized_tasks = []
        has_unsafe_tasks = False
        
        for task in tasks:
            if task.lower() == "mmlu":
                normalized_tasks.append("mmlu")
            elif task.lower() == "humaneval":
                normalized_tasks.append("humaneval")
                has_unsafe_tasks = True
            elif task.lower() == "gsm8k":
                normalized_tasks.append("gsm8k")
            elif task.lower() == "arc_challenge":
                normalized_tasks.append("arc_challenge")
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
        
        # 按 num_fewshot 分组评测 - 支持 None 值
        fewshot_groups = {}
        for task in normalized_tasks:
            fewshot = task_defaults.get(task, {}).get("num_fewshot", 0)
            if fewshot not in fewshot_groups:
                fewshot_groups[fewshot] = []
            fewshot_groups[fewshot].append(task)
        
        print(f"🎯 按 few-shot 分组:")
        for fewshot, group_tasks in fewshot_groups.items():
            fewshot_display = "默认值" if fewshot is None else f"{fewshot}-shot"
            print(f"   {fewshot_display}: {', '.join(group_tasks)}")
        
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
            "log_samples": True
        }
        
        # 只有当 num_fewshot 不为 None 时才传入参数
        if num_fewshot is not None:
            eval_args["num_fewshot"] = num_fewshot
            print(f"🎯 设置 num_fewshot={num_fewshot}")
        else:
            print(f"🎯 使用任务默认的 num_fewshot 值")
        
        # GSM8K 特殊配置
        if any(task == "gsm8k" for task in tasks):
            print(f"🧮 GSM8K 特殊配置：启用数学推理优化")
            eval_args["limit"] = None  # 评测全部样本
        
        # ARC Challenge 特殊配置
        if any(task == "arc_challenge" for task in tasks):
            print(f"🏆 ARC Challenge 特殊配置：使用默认 few-shot 设置")
        
        if has_unsafe_tasks:
            eval_args["confirm_run_unsafe_code"] = True
            print(f"🔐 安全设置：已启用 confirm_run_unsafe_code=True")
        
        # 合并用户参数
        for key, value in eval_kwargs.items():
            if key not in ['evaluation_script', 'python_executable']:
                eval_args[key] = value
        
        fewshot_display = "默认值" if num_fewshot is None else f"num_fewshot={num_fewshot}"
        print(f"⏳ 开始评测 {len(tasks)} 个任务 ({fewshot_display})...")
        results = simple_evaluate(**eval_args)
        print(f"✅ 任务组评测完成!")
        
        self.results = results
        return results

    def _evaluate_multiple_groups(self, fewshot_groups, tensor_parallel_size, 
                                 gpu_memory_utilization, has_unsafe_tasks, **eval_kwargs):
        """评测多个 few-shot 组并合并结果"""
        all_results = {"results": {}, "samples": {}}
        
        for fewshot, group_tasks in fewshot_groups.items():
            fewshot_display = "默认值" if fewshot is None else f"{fewshot}-shot"
            print(f"\n🎯 评测 {fewshot_display} 组: {', '.join(group_tasks)}")
            
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
    
    def save_results(self, output_path):
        """保存每个任务的结果到单独的文件"""
        if self.results is None:
            print("⚠️  没有评测结果可保存")
            return
        
        try:
            # 确保输出目录存在
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 保存结果到目录: {output_dir}")
            
            # 处理 samples 数据
            samples = None
            results_copy = self.results.copy()
            if self.log_samples and "samples" in results_copy:
                samples = results_copy.pop("samples")
            
            # 添加时间戳和元信息
            from datetime import datetime
            date_id = datetime.now().isoformat().replace(":", "-")
            
            # 获取评测结果
            task_results = results_copy.get("results", {})
            
            # 为每个任务保存单独的文件
            for task_name, task_result in task_results.items():
                print(f"💾 保存任务 {task_name} 的结果...")
                
                # 计算任务哈希值
                task_hash = ""
                if samples and task_name in samples:
                    task_samples = samples[task_name]
                    sample_hashes = [
                        s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                        for s in task_samples
                    ]
                    task_hash = hash_string("".join(sample_hashes))
                
                # 构建单个任务的结果字典
                single_task_result = {
                    "results": {task_name: task_result},
                    "task_hashes": {task_name: task_hash},
                    "evaluation_time": date_id,
                    "evaluation_mode": "multi_task_single_load",
                    "task_name": task_name,
                    "lora_path": self.lora_path,
                    "base_model": self.base_model_name
                }
                
                # 复制其他元数据（排除results和samples）
                for key, value in results_copy.items():
                    if key not in ["results", "task_hashes"]:
                        single_task_result[key] = value
                
                # 保存主要结果文件
                results_file = output_dir / f"lm_eval_{task_name}_results.json"
                dumped = json.dumps(
                    single_task_result, 
                    indent=2, 
                    default=handle_non_serializable, 
                    ensure_ascii=False
                )
                
                with open(results_file, "w", encoding="utf-8") as f:
                    f.write(dumped)
                print(f"✅ {task_name} 结果已保存到: {results_file}")
                
                # 保存样本数据（如果存在）
                if samples and task_name in samples:
                    samples_file = output_dir / f"lm_eval_{task_name}_results_samples.json"
                    task_samples = {task_name: samples[task_name]}
                    samples_dumped = json.dumps(
                        task_samples, 
                        indent=2, 
                        default=handle_non_serializable, 
                        ensure_ascii=False
                    )
                    with open(samples_file, "w", encoding="utf-8") as f:
                        f.write(samples_dumped)
                    print(f"✅ {task_name} 样本数据已保存到: {samples_file}")
            
            print(f"✅ 所有任务结果已保存完毕")
            
            # 打印结果表格
            print(f"\n📊 详细评测结果:")
            print(make_table(self.results))
            
        except Exception as e:
            print(f"❌ 保存结果时出错: {e}")
            import traceback
            traceback.print_exc()
            

    def run_full_pipeline(self, tasks=["mmlu"], output_path=None, **eval_kwargs):
        """运行完整的多任务评测流程：合并-评测-清理"""
        try:
            print(f"🚀 启动多任务评测流程")
            print(f"📊 任务列表: {', '.join(tasks)} (共{len(tasks)}个)")
            
            # 步骤1-4: 合并
            self.merge_lora()
            
            # 步骤5: 多任务评测
            self.evaluate_multiple_tasks(tasks=tasks, **eval_kwargs)
            
            # 保存结果
            if output_path:
                self.save_results(output_path)
            
            return self.results
            
        finally:
            # 步骤6: 清理（无论是否出错都会执行）
            self.cleanup()

def parse_tasks(tasks_str):
    """解析任务字符串 - 修正版 + GSM8K + ARC Challenge"""
    if not tasks_str:
        return ["mmlu"]
    
    tasks = [task.strip() for task in tasks_str.split(",")]
    normalized_tasks = []
    
    for task in tasks:
        if task.lower() == "all":
            # BUG (保留mmlu用于debug)
            normalized_tasks.extend(["mmlu", "humaneval", "gsm8k", "arc_challenge", "truthfulqa_mc1", "truthfulqa_mc2"])
            # normalized_tasks.extend(["humaneval", "gsm8k", "arc_challenge", "truthfulqa_mc1", "truthfulqa_mc2"])
        elif task.lower() == "truthful":  # 简化输入
            normalized_tasks.extend(["truthfulqa_mc1", "truthfulqa_mc2"])
        elif task.lower() == "gsm8k":
            normalized_tasks.append("gsm8k")
        elif task.lower() == "arc_challenge":
            normalized_tasks.append("arc_challenge")
        else:
            normalized_tasks.append(task)
    
    return normalized_tasks

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="优化版LoRA模型评测脚本 - 支持多任务一次加载 + GSM8K + ARC Challenge")
    
    parser.add_argument("--base-model", type=str, required=True,
                        help="基础模型名称或路径")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="LoRA模型路径")
    parser.add_argument("--tasks", type=str, default="humaneval",
                        help="评测任务，支持逗号分隔多个任务，如: mmlu,humaneval,gsm8k,arc_challenge,truthfulqa")
    parser.add_argument("--output-path", type=str, required=True,
                        help="输出目录路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                        help="GPU内存使用率")
    parser.add_argument("--batch-size", type=str, default="auto",
                        help="批处理大小 (推荐使用auto)")
    
    return parser

def main():
    """主函数 - 支持多任务命令行参数 + GSM8K + ARC Challenge"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 处理任务参数
    tasks = parse_tasks(args.tasks)
    
    print("🎯 启动优化版 LoRA 模型多任务评测流程 (包含 GSM8K + ARC Challenge)")
    print(f"📁 基础模型: {args.base_model}")
    print(f"📁 LoRA 路径: {args.lora_path}")
    print(f"📊 评测任务: {', '.join(tasks)} (共{len(tasks)}个)")
    print(f"⚡ Tensor Parallel: {args.tensor_parallel_size}")
    print(f"🧠 GPU内存使用率: {args.gpu_memory_utilization}")
    print(f"💾 输出目录: {args.output_path}")
    print(f"💡 优化模式: 一次加载评测{len(tasks)}个任务")
    
    # 特别提示任务
    if "gsm8k" in tasks:
        print(f"🧮 包含数学推理任务 GSM8K (8k 数学应用题)")
    if "arc_challenge" in tasks:
        print(f"🏆 包含科学推理任务 ARC Challenge (使用默认 few-shot 设置)")
    
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
            output_path=args.output_path,
            **eval_kwargs
        )
        
        print(f"\n🎉 多任务评测流程完成！")
        print(f"✅ 成功评测了 {len(tasks)} 个任务")
        if "gsm8k" in tasks:
            print(f"🧮 GSM8K 数学推理评测已完成")
        if "arc_challenge" in tasks:
            print(f"🏆 ARC Challenge 科学推理评测已完成")
        print(f"⚡ 效率提升: 相比单任务模式快约 {len(tasks)} 倍")
        print(f"📁 结果文件保存在: {args.output_path}")
        return results
        
    except Exception as e:
        print(f"❌ 多任务评测流程失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()