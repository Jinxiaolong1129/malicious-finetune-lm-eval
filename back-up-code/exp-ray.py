#!/usr/bin/env python3
# 


import os
import ray
import time
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import logging
from datetime import datetime
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ray_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class ModelEvaluator:
    """单个模型评测器 - 每个实例占用1张GPU"""
    
    def __init__(self):
        # ✅ 正确做法：只获取信息，不修改环境变量
        self.ray_gpu_ids = ray.get_gpu_ids()
        self.cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        
        logger.info(f"✅ ModelEvaluator初始化:")
        logger.info(f"   Ray分配的GPU IDs: {self.ray_gpu_ids}")
        logger.info(f"   CUDA_VISIBLE_DEVICES: {self.cuda_visible_devices}")
        
        # 验证CUDA设置
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            logger.info(f"   PyTorch可见GPU数量: {device_count}")
            logger.info(f"   PyTorch当前设备: {current_device}")
            
            # 获取GPU信息
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"   GPU {i}: {gpu_name}")
        else:
            logger.warning("⚠️ CUDA不可用")
    
    def evaluate_model(self, 
                      base_model: str,
                      lora_path: str, 
                      task: str = "mmlu",
                      output_dir: str = "./results",
                      tensor_parallel_size: int = 1,
                      gpu_memory_utilization: float = 0.8,
                      num_fewshot: int = 0,
                      **eval_kwargs) -> Dict[str, Any]:
        """评测单个模型"""
        
        start_time = time.time()
        model_name = Path(lora_path).name
        
        try:
            logger.info(f"🚀 开始评测模型: {model_name}")
            logger.info(f"   当前CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            
            # 构建输出文件路径
            output_file = Path(output_dir) / f"{model_name}_{task}_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "/home/jin509/anaconda3/envs/malicious_finetune/bin/python", 
                "ray-run_evaluation.py",
                "--base-model", base_model,
                "--lora-path", lora_path,
                "--task", task,
                "--output", str(output_file),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--num-fewshot", str(num_fewshot)
            ]
            
            # 添加其他参数
            for key, value in eval_kwargs.items():
                if value is not None:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            logger.info(f"执行命令: {' '.join(cmd[:8])}...")  # 只显示前几个参数
            
            # ✅ 正确做法：直接继承环境变量，不手动设置CUDA_VISIBLE_DEVICES
            env = os.environ.copy()
            # 不需要手动设置CUDA_VISIBLE_DEVICES，Ray已经设置好了
            
            # 执行评测
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2小时超时
                env=env  # 直接使用继承的环境变量
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ 模型 {model_name} 评测成功 (耗时: {duration:.1f}秒)")
                
                # 尝试读取结果文件 - 查找带时间戳的文件
                results_data = None
                result_files = list(output_file.parent.glob(f"{output_file.stem}_*.json"))
                
                if result_files:
                    # 使用最新的结果文件
                    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                    try:
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                        logger.info(f"✅ 读取结果文件: {latest_file}")
                    except Exception as e:
                        logger.warning(f"无法读取结果文件 {latest_file}: {e}")
                elif output_file.exists():
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"无法读取结果文件 {output_file}: {e}")
                
                return {
                    "status": "success",
                    "model_name": model_name,
                    "lora_path": lora_path,
                    "task": task,
                    "ray_gpu_ids": self.ray_gpu_ids,
                    "cuda_visible_devices": self.cuda_visible_devices,
                    "duration": duration,
                    "output_file": str(output_file),
                    "results": results_data,
                    "stdout": result.stdout[-1000:],  # 只保留最后1000字符
                    "stderr": result.stderr[-1000:] if result.stderr else ""
                }
            else:
                logger.error(f"❌ 模型 {model_name} 评测失败")
                logger.error(f"错误输出: {result.stderr}")
                
                return {
                    "status": "failed",
                    "model_name": model_name,
                    "lora_path": lora_path,
                    "task": task,
                    "ray_gpu_ids": self.ray_gpu_ids,
                    "cuda_visible_devices": self.cuda_visible_devices,
                    "duration": duration,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ 模型 {model_name} 评测超时")
            return {
                "status": "timeout",
                "model_name": model_name,
                "lora_path": lora_path,
                "task": task,
                "ray_gpu_ids": self.ray_gpu_ids,
                "duration": time.time() - start_time,
                "error": "评测超时"
            }
        except Exception as e:
            logger.error(f"💥 模型 {model_name} 评测异常: {e}")
            return {
                "status": "error",
                "model_name": model_name,
                "lora_path": lora_path,
                "task": task,
                "ray_gpu_ids": self.ray_gpu_ids,
                "duration": time.time() - start_time,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

class RayModelEvaluationManager:
    """Ray分布式模型评测管理器"""
    
    def __init__(self, num_gpus: int = None):
        # ✅ 改进：自动检测GPU数量
        if num_gpus is None:
            # 尝试检测系统GPU数量
            try:
                import torch
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                else:
                    num_gpus = 0
                    logger.warning("系统中没有可用的CUDA GPU")
            except ImportError:
                num_gpus = 0
                logger.warning("PyTorch未安装，无法检测GPU")
        
        self.num_gpus = max(1, num_gpus)  # 至少保留1个用于CPU
        self.evaluators = []
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(num_gpus=self.num_gpus)
            logger.info(f"🚀 Ray已初始化，可用GPU数量: {self.num_gpus}")
        
        # 获取实际的Ray集群资源
        cluster_resources = ray.cluster_resources()
        available_gpus = int(cluster_resources.get('GPU', 0))
        logger.info(f"📊 Ray集群资源: {cluster_resources}")
        logger.info(f"🎮 可用GPU数量: {available_gpus}")
        
        # 创建评测器实例（数量不超过可用GPU数）
        actual_evaluators = min(self.num_gpus, available_gpus)
        for i in range(actual_evaluators):
            evaluator = ModelEvaluator.remote()
            self.evaluators.append(evaluator)
        
        logger.info(f"✅ 已创建 {len(self.evaluators)} 个ModelEvaluator实例")
    
    def find_models(self, models_dir: str) -> List[str]:
        """扫描模型文件夹，找到所有LoRA模型"""
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            raise ValueError(f"模型目录不存在: {models_dir}")
        
        # 查找包含LoRA权重的目录
        model_paths = []
        for item in models_dir.iterdir():
            if item.is_dir():
                # 检查是否包含LoRA必需文件
                has_adapter_config = (item / 'adapter_config.json').exists()
                has_adapter_model = (item / 'adapter_model.bin').exists()
                has_safetensors = any(item.glob('adapter_model*.safetensors'))
                
                if has_adapter_config and (has_adapter_model or has_safetensors):
                    model_paths.append(str(item))
        
        logger.info(f"🔍 在 {models_dir} 中找到 {len(model_paths)} 个LoRA模型:")
        for path in model_paths:
            logger.info(f"  📁 {Path(path).name}")
        
        return sorted(model_paths)
    
    def evaluate_all_models(self,
                           models_dir: str,
                           base_model: str,
                           task: str = "mmlu",
                           output_dir: str = "./ray_results",
                           tensor_parallel_size: int = 1,
                           gpu_memory_utilization: float = 0.8,
                           num_fewshot: int = 0,
                           **eval_kwargs) -> List[Dict[str, Any]]:
        """评测所有模型"""
        
        # 查找所有模型
        model_paths = self.find_models(models_dir)
        
        if not model_paths:
            logger.warning("❌ 没有找到任何LoRA模型")
            return []
        
        if not self.evaluators:
            logger.error("❌ 没有可用的评测器")
            return []
        
        logger.info(f"🎯 开始评测 {len(model_paths)} 个模型，使用 {len(self.evaluators)} 个评测器")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 提交所有任务
        pending_tasks = []
        for i, model_path in enumerate(model_paths):
            # 轮询分配评测器
            evaluator = self.evaluators[i % len(self.evaluators)]
            
            task_future = evaluator.evaluate_model.remote(
                base_model=base_model,
                lora_path=model_path,
                task=task,
                output_dir=output_dir,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                num_fewshot=num_fewshot,
                **eval_kwargs
            )
            
            pending_tasks.append((task_future, Path(model_path).name))
            logger.info(f"📤 已提交任务 {i+1}/{len(model_paths)}: {Path(model_path).name}")
        
        # 等待所有任务完成并收集结果
        results = []
        completed = 0
        total = len(pending_tasks)
        
        logger.info(f"⏳ 等待 {total} 个评测任务完成...")
        
        while pending_tasks:
            # 等待至少一个任务完成
            ready_futures = [task[0] for task in pending_tasks]
            ready_tasks, remaining_tasks = ray.wait(ready_futures, num_returns=1, timeout=60)
            
            # 处理完成的任务
            for ready_future in ready_tasks:
                # 找到对应的任务
                for i, (task_future, model_name) in enumerate(pending_tasks):
                    if task_future == ready_future:
                        try:
                            result = ray.get(ready_future)
                            results.append(result)
                            completed += 1
                            
                            status = result['status']
                            duration = result.get('duration', 0)
                            
                            if status == 'success':
                                logger.info(f"✅ [{completed}/{total}] {model_name} 评测成功 ({duration:.1f}s)")
                            else:
                                error_msg = result.get('error', 'Unknown error')[:100]
                                logger.error(f"❌ [{completed}/{total}] {model_name} 评测失败: {error_msg}")
                                
                        except Exception as e:
                            logger.error(f"💥 获取任务结果时出错: {e}")
                            completed += 1
                        
                        # 从待处理列表中移除
                        pending_tasks.pop(i)
                        break
            
            # 显示进度
            if completed > 0 and completed % max(1, total // 10) == 0:
                progress = completed / total * 100
                logger.info(f"📊 进度: {completed}/{total} ({progress:.1f}%)")
        
        # 保存汇总结果
        self.save_summary_results(results, output_dir, task)
        
        return results
    
    def save_summary_results(self, results: List[Dict[str, Any]], output_dir: str, task: str):
        """保存汇总结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = Path(output_dir) / f"evaluation_summary_{task}_{timestamp}.json"
        
        # 统计信息
        total_models = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        timeout = sum(1 for r in results if r['status'] == 'timeout')
        error = sum(1 for r in results if r['status'] == 'error')
        
        # 计算总耗时
        total_duration = sum(r.get('duration', 0) for r in results)
        avg_duration = total_duration / total_models if total_models > 0 else 0
        
        summary = {
            "evaluation_time": datetime.now().isoformat(),
            "task": task,
            "statistics": {
                "total_models": total_models,
                "successful": successful,
                "failed": failed,
                "timeout": timeout,
                "error": error,
                "success_rate": successful / total_models if total_models > 0 else 0,
                "total_duration_seconds": total_duration,
                "average_duration_seconds": avg_duration
            },
            "results": results
        }
        
        # 保存详细结果
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 汇总结果已保存到: {summary_file}")
        logger.info(f"📊 评测统计:")
        logger.info(f"   总模型数: {total_models}")
        logger.info(f"   成功: {successful} ({successful/total_models*100:.1f}%)")
        logger.info(f"   失败: {failed}")
        logger.info(f"   超时: {timeout}")
        logger.info(f"   错误: {error}")
        logger.info(f"   总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
        logger.info(f"   平均耗时: {avg_duration:.1f}秒")
        
        # 保存简化的结果表格
        self.save_results_table(results, output_dir, task)
    
    def save_results_table(self, results: List[Dict[str, Any]], output_dir: str, task: str):
        """保存结果表格"""
        table_file = Path(output_dir) / f"results_table_{task}.txt"
        
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(f"LoRA模型评测结果汇总 - 任务: {task}\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'模型名称':<40} {'状态':<10} {'Ray GPU':<10} {'CUDA设备':<12} {'耗时(秒)':<10} {'备注'}\n")
            f.write("-" * 100 + "\n")
            
            for result in results:
                model_name = result['model_name'][:38]
                status = result['status']
                ray_gpus = str(result.get('ray_gpu_ids', 'N/A'))[:8]
                cuda_devices = str(result.get('cuda_visible_devices', 'N/A'))[:10]
                duration = f"{result.get('duration', 0):.1f}"
                
                if status == 'success':
                    note = "✅"
                elif status == 'failed':
                    note = "❌ " + str(result.get('error', ''))[:20]
                elif status == 'timeout':
                    note = "⏰ 超时"
                else:
                    note = "💥 异常"
                
                f.write(f"{model_name:<40} {status:<10} {ray_gpus:<10} {cuda_devices:<12} {duration:<10} {note}\n")
        
        logger.info(f"📋 结果表格已保存到: {table_file}")
    
    def shutdown(self):
        """关闭Ray"""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("🔚 Ray已关闭")

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="Ray分布式LoRA模型评测系统")
    
    parser.add_argument("--models-dir", type=str, required=True,
                        help="包含所有LoRA模型的目录")
    parser.add_argument("--base-model", type=str, 
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="基础模型名称或路径")
    parser.add_argument("--task", type=str, default="mmlu",
                        choices=["mmlu", "humaneval", "truthfulqa", "all"],
                        help="评测任务")
    parser.add_argument("--output-dir", type=str, default="./ray_results",
                        help="结果输出目录")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="使用的GPU数量 (默认自动检测)")
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
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    logger.info("🚀 启动Ray分布式LoRA模型评测系统")
    logger.info(f"📁 模型目录: {args.models_dir}")
    logger.info(f"🤖 基础模型: {args.base_model}")
    logger.info(f"📊 评测任务: {args.task}")
    logger.info(f"💾 输出目录: {args.output_dir}")
    logger.info(f"🎮 指定GPU数量: {args.num_gpus or '自动检测'}")
    logger.info(f"⚡ Tensor Parallel: {args.tensor_parallel_size}")
    logger.info(f"🧠 GPU内存使用率: {args.gpu_memory_utilization}")
    logger.info(f"🎯 Few-shot数量: {args.num_fewshot}")
    
    # 创建管理器
    manager = RayModelEvaluationManager(num_gpus=args.num_gpus)
    
    try:
        # 构建评测参数
        eval_kwargs = {}
        if args.batch_size != "auto":
            eval_kwargs["batch_size"] = args.batch_size
        
        # 开始评测
        start_time = time.time()
        results = manager.evaluate_all_models(
            models_dir=args.models_dir,
            base_model=args.base_model,
            task=args.task,
            output_dir=args.output_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_fewshot=args.num_fewshot,
            **eval_kwargs
        )
        
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if r['status'] == 'success')
        total = len(results)
        
        logger.info(f"🎉 评测完成！")
        logger.info(f"📊 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        logger.info(f"✅ 成功率: {successful}/{total} ({successful/total*100:.1f}%)")
        logger.info(f"📁 结果保存在: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ 评测过程中出现错误: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        manager.shutdown()

if __name__ == "__main__":
    main()