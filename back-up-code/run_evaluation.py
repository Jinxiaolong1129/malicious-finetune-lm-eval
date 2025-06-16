#!/usr/bin/env python3
# run_evaluation.py


import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


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
    

    
    def evaluate(self, tasks=["mmlu"], **eval_kwargs):
        """使用 vLLM 评测合并后的模型"""
        if not self.merged_path or not os.path.exists(self.merged_path):
            raise ValueError("请先调用 merge_lora() 合并模型")
        
        print(f"\n🚀 步骤5: 使用 vLLM 开始MMLU评测...")
        print(f"📊 评测任务: {', '.join(tasks)}")
        
        # 默认的 vLLM 参数
        default_model_args = {
            "pretrained": self.merged_path,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 256,            
            "max_num_batched_tokens": 4096, 
        }
        
        # 合并用户提供的参数
        model_args = default_model_args.copy()
        if "model_args" in eval_kwargs:
            model_args.update(eval_kwargs.pop("model_args"))
        
        # 默认的评测参数 - 专为MMLU配置
        default_eval_args = {
            "model": "vllm",
            "model_args": model_args,
            "tasks": tasks,
            "batch_size": "auto",
            "num_fewshot": 0, 
            "log_samples": True
        }
        
        # 合并用户提供的参数
        final_eval_args = default_eval_args.copy()
        final_eval_args.update(eval_kwargs)
        
        final_eval_args["num_fewshot"] = 0
        
        try:
            self.results = simple_evaluate(**final_eval_args)
            print("✅ MMLU评测完成!")
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
            
            # # 处理 samples 数据
            samples = None
            results_copy = self.results.copy()
            if self.log_samples and "samples" in results_copy:
                samples = results_copy.pop("samples")
        
            # 序列化结果
            dumped = json.dumps(
                self.results, 
                indent=2, 
                default=handle_non_serializable, 
                ensure_ascii=False
            )
            
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 添加时间戳到文件名
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
            
            
    def print_summary(self):
        """打印MMLU评测结果摘要"""
        if not self.results:
            print("⚠️  没有评测结果")
            return
        
        print(f"\n{'='*60}")
        print("📊 MMLU评测结果摘要")
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
        """运行完整的MMLU评测流程：合并-评测-清理"""
        try:
            # 步骤1-4: 合并
            self.merge_lora()
            
            # 步骤5: 评测（默认使用MMLU）
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

def main():
    """主函数 - 使用您的具体路径"""
    
    # 配置参数
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    lora_path = "/data3/user/jin509/malicious-finetuning/experiments/default/gsm8k-BeaverTails-p0.2/Llama-2-7b-chat-hf-lora-r64-e10-b16-data100"
    
    # MMLU评测 - 使用默认的mmlu任务，5-shot
    tasks = ["mmlu"]  # 使用默认的mmlu任务
    
    # 输出文件
    output_file = "mmlu_evaluation_results.json"
    
    print("🎯 开始 LoRA 模型 MMLU 自动化评测流程")
    print(f"📁 基础模型: {base_model_name}")
    print(f"📁 LoRA 路径: {lora_path}")
    print(f"📊 评测数据集: MMLU")
    print(f"🎯 评测模式: 5-shot")
    print(f"💾 输出文件: {output_file}")
    print(f"{'='*60}")
    
    # 创建评测器
    evaluator = LoRAEvaluator(base_model_name, lora_path)
    
    # 运行完整流程
    try:
        results = evaluator.run_full_pipeline(
            tasks=tasks,
            output_file=output_file,
        )
        
        print(f"\n🎉 MMLU评测流程完成！")
        return results
        
    except Exception as e:
        print(f"❌ 评测流程失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()