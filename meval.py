#!/usr/bin/env python3

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time

import shutil
import json
import tempfile
import argparse
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
    
    # 支持的评测任务配置
    SUPPORTED_TASKS = {
        "mmlu": {
            "tasks": ["mmlu"],
            "num_fewshot": 5,
            "description": "MMLU (Massive Multitask Language Understanding) - 大规模多任务语言理解"
        },
        "humaneval": {
            "tasks": ["humaneval"],
            "num_fewshot": 0,
            "description": "HumanEval - 代码生成能力评测"
        },
        "truthfulqa": {
            "tasks": ["truthfulqa_mc1", "truthfulqa_mc2"],
            "num_fewshot": 0,
            "description": "TruthfulQA - 真实性问答评测"
        },
        "gpqa": {
            "tasks": ["gpqa"],
            "num_fewshot": 0,  # 官方默认零样本评测
            "description": "GPQA (Graduate-Level Google-Proof Q&A) - 研究生级别问答评测 (~448题, 0-shot)"
        },
        "commonsense_qa": {
            "tasks": ["commonsenseqa"],
            "num_fewshot": 0,  # 官方默认零样本评测
            "description": "CommonsenseQA - 常识推理问答评测 (~1,221题验证集, 0-shot)"
        },
        "winogrande": {
            "tasks": ["winogrande"],
            "num_fewshot": 0,  # 官方默认零样本评测
            "description": "WinoGrande - 代词消歧常识推理评测 (~1,767题测试集, 0-shot)"
        },
        "reasoning": {
            "tasks": ["gpqa", "commonsense_qa", "winogrande"],
            "num_fewshot": 0,  # 保持与单独任务一致
            "description": "推理能力综合评测 (GPQA + CommonsenseQA + WinoGrande, 0-shot)"
        },
        "all": {
            "tasks": ["mmlu", "humaneval", "truthfulqa_mc1", "truthfulqa_mc2", "gpqa", "commonsenseqa", "winogrande"],
            "num_fewshot": "auto",  # 每个任务使用默认值
            "description": "所有评测任务"
        }
    }
    
    def __init__(self, base_model_name, lora_path):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.merged_path = None
        self.results = None
        
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
    
    def get_task_config(self, task_name):
        """获取任务配置"""
        if task_name not in self.SUPPORTED_TASKS:
            raise ValueError(f"不支持的任务: {task_name}。支持的任务: {list(self.SUPPORTED_TASKS.keys())}")
        return self.SUPPORTED_TASKS[task_name]
    
    def evaluate(self, task_name="mmlu", **eval_kwargs):
        """使用 vLLM 评测合并后的模型"""
        if not self.merged_path or not os.path.exists(self.merged_path):
            raise ValueError("请先调用 merge_lora() 合并模型")
        
        # 获取任务配置
        task_config = self.get_task_config(task_name)
        tasks = task_config["tasks"]
        default_fewshot = task_config["num_fewshot"]
        
        print(f"\n🚀 步骤5: 使用 vLLM 开始评测...")
        print(f"📊 评测任务: {task_config['description']}")
        print(f"🎯 具体任务: {', '.join(tasks)}")
        
        # 根据任务类型优化batch_size
        if task_name in ["gpqa", "commonsenseqa", "winogrande", "reasoning"]:
            # 这些任务较小，可以使用较大的batch_size
            default_batch_size = 32
            max_num_seqs = 512
        else:
            # 大任务使用较小的batch_size
            default_batch_size = 16  
            max_num_seqs = 256
        
        # 默认的 vLLM 参数
        default_model_args = {
            "pretrained": self.merged_path,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": max_num_seqs,            
            "max_num_batched_tokens": 4096, 
        }
        
        # 合并用户提供的参数
        model_args = default_model_args.copy()
        if "model_args" in eval_kwargs:
            model_args.update(eval_kwargs.pop("model_args"))
        
        # 确定 few-shot 数量
        if "num_fewshot" not in eval_kwargs:
            if default_fewshot == "auto":
                # 对于 "all" 任务，使用任务特定的默认值
                eval_kwargs["num_fewshot"] = None  # 让 lm_eval 使用默认值
            else:
                eval_kwargs["num_fewshot"] = default_fewshot
        
        # 默认的评测参数
        default_eval_args = {
            "model": "vllm",
            "model_args": model_args,
            "tasks": tasks,
            "batch_size": eval_kwargs.get("batch_size", default_batch_size),
            "log_samples": True,
            'confirm_run_unsafe_code': True,
            # "limit": 10,  # 只评测前10个样本，调试时可以启用
        }
        
        # 合并用户提供的参数
        final_eval_args = default_eval_args.copy()
        final_eval_args.update(eval_kwargs)
        
        # 显示few-shot信息
        fewshot_info = final_eval_args.get("num_fewshot", "默认")
        print(f"🎯 Few-shot 模式: {fewshot_info}")
        print(f"🚀 批处理大小: {final_eval_args['batch_size']}")
        
        try:
            self.results = simple_evaluate(**final_eval_args)
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
        self.log_samples = True
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
            final_output_file = output_path
            final_output_file = output_path
            
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
            

    def print_summary(self, task_name):
        """打印评测结果摘要"""
        if not self.results:
            print("⚠️  没有评测结果")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 {self.SUPPORTED_TASKS[task_name]['description']} 评测结果摘要")
        print(f"{'='*70}")
        
        # 收集所有指标
        all_metrics = []
        task_results = []
        
        for task_name_full, task_results_dict in self.results.get("results", {}).items():
            # 根据不同任务类型显示不同指标
            if "mmlu" in task_name_full:
                metric = task_results_dict.get("acc", 0.0)
                metric_name = "准确率"
            elif "humaneval" in task_name_full:
                metric = task_results_dict.get("pass@1", 0.0)
                metric_name = "Pass@1"
            elif "hellaswag" in task_name_full:
                metric = task_results_dict.get("acc_norm", task_results_dict.get("acc", 0.0))
                metric_name = "标准化准确率"
            elif "truthfulqa" in task_name_full:
                if "mc1" in task_name_full:
                    metric = task_results_dict.get("acc", 0.0)
                    metric_name = "MC1准确率"
                else:
                    metric = task_results_dict.get("acc", 0.0)
                    metric_name = "MC2准确率"
            elif "gpqa" in task_name_full:
                metric = task_results_dict.get("acc", 0.0)
                metric_name = "准确率"
            elif "commonsenseqa" in task_name_full:
                metric = task_results_dict.get("acc", 0.0)
                metric_name = "准确率"
            elif "winogrande" in task_name_full:
                metric = task_results_dict.get("acc", 0.0)
                metric_name = "准确率"
            else:
                metric = task_results_dict.get("acc", 0.0)
                metric_name = "准确率"
            
            all_metrics.append(metric)
            task_results.append((task_name_full, metric, metric_name))
        
        # 按指标值排序
        task_results.sort(key=lambda x: x[1], reverse=True)
        
        # 计算总体统计
        if all_metrics:
            avg_metric = sum(all_metrics) / len(all_metrics)
            print(f"\n🎯 总体表现:")
            print(f"  平均指标值: {avg_metric:.4f}")
            print(f"  最高指标值: {max(all_metrics):.4f}")
            print(f"  最低指标值: {min(all_metrics):.4f}")
            print(f"  评测任务数: {len(all_metrics)}")
        
        # 显示各任务详细结果
        print(f"\n📋 各任务详细结果 (按指标值排序):")
        print("-" * 70)
        for task_name_full, metric, metric_name in task_results:
            # 简化显示名称
            display_name = task_name_full.replace("mmlu_", "").replace("truthfulqa_", "")
            print(f"  {display_name:<35}: {metric:.4f} ({metric_name})")
    
    def run_full_pipeline(self, task_name="mmlu", output_file=None, **eval_kwargs):
        """运行完整的评测流程：合并-评测-清理"""
        try:
            # 验证任务名称
            if task_name not in self.SUPPORTED_TASKS:
                raise ValueError(f"不支持的任务: {task_name}。支持的任务: {list(self.SUPPORTED_TASKS.keys())}")
            
            # 步骤1-4: 合并
            self.merge_lora()
            
            # 步骤5: 评测
            self.evaluate(task_name=task_name, **eval_kwargs)
            
            # 保存结果
            if output_file:
                self.save_results(output_file)
            
            # 打印摘要
            self.print_summary(task_name)
            
            return self.results
            
        finally:
            # 步骤6: 清理（无论是否出错都会执行）
            self.cleanup()

def create_parser():
    """创建命令行参数解析器"""
    # 配置默认参数
    DEFAULT_BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
    DEFAULT_LORA_PATH = "/data3/user/jin509/malicious-finetuning/experiments/default/gsm8k-BeaverTails-p0.2/Llama-2-7b-chat-hf-lora-r64-e10-b16-data100"
    
    parser = argparse.ArgumentParser(description="LoRA 模型多任务评测工具")
    
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"基础模型名称或路径 (默认: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--lora-path", type=str, default=DEFAULT_LORA_PATH,
                        help=f"LoRA 适配器路径 (默认: {DEFAULT_LORA_PATH})")
    parser.add_argument("--task", type=str, default="mmlu",
                        choices=list(LoRAEvaluator.SUPPORTED_TASKS.keys()),
                        help="评测任务 (默认: mmlu)")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件路径 (默认: 根据任务自动生成)")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Few-shot 数量 (默认: 使用任务推荐值)")
    parser.add_argument("--batch-size", type=str, default="auto",
                        help="批处理大小 (默认: auto)")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="张量并行大小 (默认: 4)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="GPU 内存利用率 (默认: 0.8)")
    
    return parser

def main():
    """主函数 - 支持命令行参数和默认运行"""
    
    # 配置默认参数
    DEFAULT_BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
    DEFAULT_LORA_PATH = "/data3/user/jin509/malicious-finetuning/experiments/default/gsm8k-BeaverTails-p0.2/Llama-2-7b-chat-hf-lora-r64-e10-b16-data100"
    
    parser = create_parser()
    
    # 如果没有命令行参数，使用默认值
    import sys
    if len(sys.argv) == 1:
        print("🔧 使用默认配置运行...")
        
        # 默认配置
        base_model_name = DEFAULT_BASE_MODEL
        lora_path = DEFAULT_LORA_PATH
        task_name = "mmlu"
        output_file = f"{task_name}_evaluation_results.json"
        
        eval_kwargs = {}
        
    else:
        # 解析命令行参数
        args = parser.parse_args()
        
        base_model_name = args.base_model
        lora_path = args.lora_path
        task_name = args.task
        
        # 自动生成输出文件名
        if args.output is None:
            output_file = f"{task_name}_evaluation_results.json"
        else:
            output_file = args.output
        
        # 构建评测参数
        eval_kwargs = {
            "batch_size": args.batch_size,
            "model_args": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        }
        
        if args.num_fewshot is not None:
            eval_kwargs["num_fewshot"] = args.num_fewshot
    
    # 显示配置信息
    evaluator = LoRAEvaluator(base_model_name, lora_path)
    task_config = evaluator.get_task_config(task_name)
    
    print(f'start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print("🎯 开始 LoRA 模型多任务自动化评测流程")
    print(f"📁 基础模型: {base_model_name}")
    print(f"📁 LoRA 路径: {lora_path}")
    print(f"📊 评测任务: {task_config['description']}")
    print(f"🎯 具体任务: {', '.join(task_config['tasks'])}")
    print(f"💾 输出文件: {output_file}")
    print(f"🔧 Few-shot: {eval_kwargs.get('num_fewshot', '任务默认值')}")
    print(f"📝 使用{'默认' if len(sys.argv) == 1 else '命令行'}配置")
    print(f"{'='*70}")
    
    # 显示所有支持的任务
    print("\n📋 支持的评测任务:")
    for task, config in evaluator.SUPPORTED_TASKS.items():
        print(f"  • {task}: {config['description']}")
    print()
    
    # 运行完整流程
    try:
        results = evaluator.run_full_pipeline(
            task_name=task_name,
            output_file=output_file,
            **eval_kwargs
        )
        
        print(f"\n🎉 {task_config['description']} 评测流程完成！")
        print(f'end time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        return results
        
    except Exception as e:
        print(f"❌ 评测流程失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()