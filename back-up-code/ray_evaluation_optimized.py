#!/usr/bin/env python3
# 优化版Ray分布式LoRA模型评测系统 - 一次加载多任务评测 + 日志保存

import os
import ray
import time
import json
import torch
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union
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
class OptimizedModelEvaluator:
    """优化的模型评测器 - 一次加载多任务评测"""
    
    def __init__(self):
        self.ray_gpu_ids = ray.get_gpu_ids()
        self.cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        
        logger.info(f"✅ OptimizedModelEvaluator初始化:")
        logger.info(f"   Ray分配的GPU IDs: {self.ray_gpu_ids}")
        logger.info(f"   CUDA_VISIBLE_DEVICES: {self.cuda_visible_devices}")
        
        # 验证CUDA设置
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            logger.info(f"   PyTorch可见GPU数量: {device_count}")
            logger.info(f"   PyTorch当前设备: {current_device}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"   GPU {i}: {gpu_name}")
        else:
            logger.warning("⚠️ CUDA不可用")
    
    def evaluate_model_multi_tasks(self, 
                                  base_model: str,
                                  lora_path: str, 
                                  tasks: List[str] = ["mmlu"],
                                  output_dir: str = "./results",
                                  evaluation_script: str = "ray-run_evaluation.py",
                                  python_executable: str = None,
                                  **eval_kwargs) -> Dict[str, Any]:
        """评测单个模型的多个任务 - 一次加载"""
        
        start_time = time.time()
        model_name = Path(lora_path).name
        
        try:
            logger.info(f"🚀 开始评测模型: {model_name}")
            logger.info(f"📊 评测任务: {', '.join(tasks)}")
            logger.info(f"   当前CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            
            # 构建输出文件路径 - 使用时间戳避免冲突
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 精确到毫秒
            output_file = Path(output_dir) / f"{model_name}_multi_tasks_{timestamp}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建日志目录
            log_dir = Path(output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 为每个评测进程创建专用日志文件
            log_file_path = log_dir / f"{model_name}_eval_{timestamp}.log"
            
            # 使用配置中的Python可执行文件路径
            if python_executable is None:
                python_executable = "/home/jin509/anaconda3/envs/malicious_finetune/bin/python"
            
            # 构建命令 - 传递所有任务给单个脚本调用
            tasks_str = ",".join(tasks)  # 用逗号分隔的任务列表
            
            cmd = [
                python_executable,
                evaluation_script,
                "--base-model", base_model,
                "--lora-path", lora_path,
                "--tasks", tasks_str,  # 改为复数形式，传递多个任务
                "--output", str(output_file)
            ]
            
            # 添加其他评测参数
            for key, value in eval_kwargs.items():
                if value is not None and key not in ['evaluation_script', 'python_executable']:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            logger.info(f"执行命令: {' '.join(cmd[:8])}...")
            logger.info(f"📝 日志文件: {log_file_path}")
            
            # 直接继承环境变量
            env = os.environ.copy()
            
            # 执行评测 - 保存输出到日志文件
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                # 写入评测开始信息
                log_file.write(f"=== 开始评测 {model_name} ===\n")
                log_file.write(f"时间: {datetime.now().isoformat()}\n")
                log_file.write(f"模型路径: {lora_path}\n")
                log_file.write(f"基础模型: {base_model}\n")
                log_file.write(f"评测任务: {', '.join(tasks)}\n")
                log_file.write(f"命令: {' '.join(cmd)}\n")
                log_file.write(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
                log_file.write(f"Ray GPU IDs: {self.ray_gpu_ids}\n")
                log_file.write("=" * 80 + "\n\n")
                log_file.flush()
                
                # 启动子进程，将stdout和stderr都重定向到日志文件
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # 将stderr合并到stdout
                    text=True,
                    timeout=14400,  # 4小时超时（多任务需要更长时间）
                    env=env
                )
                
                # 写入评测结束信息
                log_file.write(f"\n\n=== 评测结束 ===\n")
                log_file.write(f"结束时间: {datetime.now().isoformat()}\n")
                log_file.write(f"返回码: {result.returncode}\n")
                log_file.write(f"耗时: {time.time() - start_time:.2f} 秒\n")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 读取保存的日志文件内容（用于返回结果）
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                # 只保留最后2000个字符作为摘要
                log_summary = log_content[-2000:] if len(log_content) > 2000 else log_content
            except Exception as e:
                logger.warning(f"无法读取日志文件 {log_file_path}: {e}")
                log_summary = "日志文件读取失败"
            
            if result.returncode == 0:
                logger.info(f"✅ 模型 {model_name} 多任务评测成功 (耗时: {duration:.1f}秒)")
                
                # 尝试读取结果文件
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
                    "tasks": tasks,
                    "task_count": len(tasks),
                    "ray_gpu_ids": self.ray_gpu_ids,
                    "cuda_visible_devices": self.cuda_visible_devices,
                    "duration": duration,
                    "output_file": str(output_file),
                    "log_file": str(log_file_path),  # 新增：日志文件路径
                    "results": results_data,
                    "log_summary": log_summary,  # 新增：日志摘要
                    "returncode": result.returncode
                }
            else:
                logger.error(f"❌ 模型 {model_name} 多任务评测失败")
                logger.error(f"日志文件: {log_file_path}")
                
                return {
                    "status": "failed",
                    "model_name": model_name,
                    "lora_path": lora_path,
                    "tasks": tasks,
                    "task_count": len(tasks),
                    "ray_gpu_ids": self.ray_gpu_ids,
                    "cuda_visible_devices": self.cuda_visible_devices,
                    "duration": duration,
                    "log_file": str(log_file_path),  # 新增：日志文件路径
                    "log_summary": log_summary,  # 新增：日志摘要
                    "returncode": result.returncode,
                    "error": f"评测失败，详细信息请查看日志文件: {log_file_path}"
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ 模型 {model_name} 多任务评测超时")
            
            # 超时时也要记录日志
            try:
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n\n=== 评测超时 ===\n")
                    log_file.write(f"超时时间: {datetime.now().isoformat()}\n")
                    log_file.write(f"超时设置: 14400 秒 (4小时)\n")
                    log_file.write(f"实际耗时: {time.time() - start_time:.2f} 秒\n")
            except:
                pass
                
            return {
                "status": "timeout",
                "model_name": model_name,
                "lora_path": lora_path,
                "tasks": tasks,
                "task_count": len(tasks),
                "ray_gpu_ids": self.ray_gpu_ids,
                "duration": time.time() - start_time,
                "log_file": str(log_file_path) if 'log_file_path' in locals() else None,
                "error": "评测超时 (4小时)"
            }
        except Exception as e:
            logger.error(f"💥 模型 {model_name} 多任务评测异常: {e}")
            
            # 异常时也要记录日志
            try:
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n\n=== 评测异常 ===\n")
                    log_file.write(f"异常时间: {datetime.now().isoformat()}\n")
                    log_file.write(f"异常信息: {str(e)}\n")
                    log_file.write(f"异常堆栈:\n{traceback.format_exc()}\n")
            except:
                pass
                
            return {
                "status": "error",
                "model_name": model_name,
                "lora_path": lora_path,
                "tasks": tasks,
                "task_count": len(tasks),
                "ray_gpu_ids": self.ray_gpu_ids,
                "duration": time.time() - start_time,
                "log_file": str(log_file_path) if 'log_file_path' in locals() else None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 无法加载配置文件 {config_path}: {e}")
            raise
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """验证配置文件格式"""
        required_sections = ['models', 'evaluation']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ 配置文件缺少必需部分: {section}")
                return False
        
        if not isinstance(config['models'], dict):
            logger.error("❌ 'models' 部分应该是字典格式")
            return False
        
        if not isinstance(config['evaluation'], dict):
            logger.error("❌ 'evaluation' 部分应该是字典格式")
            return False
        
        logger.info("✅ 配置文件格式验证通过")
        return True
    
    @staticmethod
    def resolve_model_paths(config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析模型路径"""
        model_list = []
        models_config = config['models']
        
        base_model = models_config.get('base_model', 'meta-llama/Llama-2-7b-chat-hf')
        lora_models = models_config.get('lora_models', [])
        
        for model_entry in lora_models:
            if isinstance(model_entry, str):
                model_list.append({
                    'name': Path(model_entry).name,
                    'path': model_entry,
                    'base_model': base_model
                })
            elif isinstance(model_entry, dict):
                model_path = model_entry.get('path')
                model_name = model_entry.get('name', Path(model_path).name if model_path else 'unknown')
                model_base = model_entry.get('base_model', base_model)
                
                if model_path:
                    model_list.append({
                        'name': model_name,
                        'path': model_path,
                        'base_model': model_base
                    })
        
        # 处理目录扫描
        if 'scan_directories' in models_config:
            scan_dirs = models_config['scan_directories']
            if not isinstance(scan_dirs, list):
                scan_dirs = [scan_dirs]
            
            for scan_dir in scan_dirs:
                scanned_models = ConfigManager._scan_directory_for_models(scan_dir, base_model)
                model_list.extend(scanned_models)
        
        logger.info(f"🔍 解析得到 {len(model_list)} 个模型:")
        for model in model_list:
            logger.info(f"  📁 {model['name']} -> {model['path']}")
        
        return model_list
    
    @staticmethod
    def _scan_directory_for_models(directory: str, base_model: str) -> List[Dict[str, Any]]:
        """扫描目录中的LoRA模型"""
        models = []
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"⚠️ 扫描目录不存在: {directory}")
            return models
        
        for item in directory.iterdir():
            if item.is_dir():
                has_adapter_config = (item / 'adapter_config.json').exists()
                has_adapter_model = (item / 'adapter_model.bin').exists()
                has_safetensors = any(item.glob('adapter_model*.safetensors'))
                
                if has_adapter_config and (has_adapter_model or has_safetensors):
                    models.append({
                        'name': item.name,
                        'path': str(item),
                        'base_model': base_model
                    })
        
        return models

class OptimizedRayModelEvaluationManager:
    """优化的Ray分布式模型评测管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 从配置中获取Ray设置
        ray_config = config.get('ray', {})
        num_gpus = ray_config.get('num_gpus')
        
        # 自动检测GPU数量
        if num_gpus is None:
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
        
        self.num_gpus = max(1, num_gpus)
        self.evaluators = []
        
        # 初始化Ray
        if not ray.is_initialized():
            ray_init_config = ray_config.get('init_config', {})
            ray.init(num_gpus=self.num_gpus, **ray_init_config)
            logger.info(f"🚀 Ray已初始化，可用GPU数量: {self.num_gpus}")
        
        # 获取实际的Ray集群资源
        cluster_resources = ray.cluster_resources()
        available_gpus = int(cluster_resources.get('GPU', 0))
        logger.info(f"📊 Ray集群资源: {cluster_resources}")
        logger.info(f"🎮 可用GPU数量: {available_gpus}")
        
        # 创建评测器实例
        actual_evaluators = min(self.num_gpus, available_gpus)
        for i in range(actual_evaluators):
            evaluator = OptimizedModelEvaluator.remote()
            self.evaluators.append(evaluator)
        
        logger.info(f"✅ 已创建 {len(self.evaluators)} 个OptimizedModelEvaluator实例")
    
    def evaluate_all_models(self) -> List[Dict[str, Any]]:
        """评测所有配置的模型 - 优化版：每个模型一次加载评测所有任务"""
        
        # 解析模型列表
        model_list = ConfigManager.resolve_model_paths(self.config)
        
        if not model_list:
            logger.warning("❌ 没有找到任何模型")
            return []
        
        if not self.evaluators:
            logger.error("❌ 没有可用的评测器")
            return []
        
        # 获取评测配置
        eval_config = self.config['evaluation']
        tasks = eval_config.get('tasks', ['mmlu'])
        if isinstance(tasks, str):
            tasks = [tasks]
        
        output_dir = eval_config.get('output_dir', './ray_results')
        
        # 创建输出目录和日志目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 结果目录: {output_dir}")
        logger.info(f"📝 日志目录: {log_dir}")
        
        # 为每个模型创建一个评测任务（包含所有任务）
        all_tasks = []
        task_info = []
        
        for i, model_info in enumerate(model_list):
            # 准备评测参数
            eval_params = {
                'base_model': model_info['base_model'],
                'lora_path': model_info['path'],
                'tasks': tasks,  # 传递所有任务
                'output_dir': output_dir,
                **eval_config.get('parameters', {})
            }
            
            # 轮询分配评测器
            evaluator = self.evaluators[i % len(self.evaluators)]
            
            task_future = evaluator.evaluate_model_multi_tasks.remote(**eval_params)
            all_tasks.append(task_future)
            task_info.append({
                'model_name': model_info['name'],
                'tasks': tasks,
                'task_count': len(tasks),
                'path': model_info['path']
            })
        
        total_models = len(model_list)
        total_task_count = len(tasks) * total_models
        
        logger.info(f"🎯 优化模式：每个模型一次加载评测所有任务")
        logger.info(f"📊 评测 {total_models} 个模型，每个模型 {len(tasks)} 个任务")
        logger.info(f"🔧 使用 {len(self.evaluators)} 个评测器")
        logger.info(f"⚡ 总任务数: {total_task_count}，但只需 {total_models} 次模型加载")
        logger.info(f"📝 每个评测进程的日志将保存在 {log_dir}")
        
        # 等待所有任务完成并收集结果
        results = []
        completed = 0
        
        logger.info(f"⏳ 等待 {total_models} 个模型评测完成...")
        
        start_time = time.time()
        
        while all_tasks:
            # 等待至少一个任务完成
            ready_tasks, remaining_tasks = ray.wait(all_tasks, num_returns=1, timeout=60)
            
            # 处理完成的任务
            for ready_future in ready_tasks:
                # 找到对应的任务信息
                task_index = all_tasks.index(ready_future)
                info = task_info[task_index]
                
                try:
                    result = ray.get(ready_future)
                    results.append(result)
                    completed += 1
                    
                    status = result['status']
                    duration = result.get('duration', 0)
                    task_count = result.get('task_count', 0)
                    log_file = result.get('log_file', 'N/A')
                    
                    if status == 'success':
                        logger.info(f"✅ [{completed}/{total_models}] {info['model_name']} "
                                  f"({task_count}个任务) 评测成功 ({duration:.1f}s)")
                        logger.info(f"   📝 日志: {log_file}")
                    else:
                        error_msg = result.get('error', 'Unknown error')[:100]
                        logger.error(f"❌ [{completed}/{total_models}] {info['model_name']} "
                                   f"({task_count}个任务) 评测失败: {error_msg}")
                        logger.error(f"   📝 日志: {log_file}")
                        
                except Exception as e:
                    logger.error(f"💥 获取任务结果时出错: {e}")
                    completed += 1
                
                # 从列表中移除已完成的任务
                all_tasks.remove(ready_future)
                task_info.pop(task_index)
            
            # 显示进度
            if completed > 0 and completed % max(1, total_models // 10) == 0:
                elapsed = time.time() - start_time
                progress = completed / total_models * 100
                eta = (elapsed / completed) * (total_models - completed) if completed > 0 else 0
                logger.info(f"📊 进度: {completed}/{total_models} ({progress:.1f}%) - "
                          f"已用时: {elapsed/60:.1f}min - 预计剩余: {eta/60:.1f}min")
        
        # 保存汇总结果
        self.save_summary_results(results, output_dir)
        
        return results
    
    def save_summary_results(self, results: List[Dict[str, Any]], output_dir: str):
        """保存汇总结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = Path(output_dir) / f"evaluation_summary_{timestamp}.json"
        
        # 统计信息
        total_models = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        timeout = sum(1 for r in results if r['status'] == 'timeout')
        error = sum(1 for r in results if r['status'] == 'error')
        
        # 计算总耗时和任务统计
        total_duration = sum(r.get('duration', 0) for r in results)
        avg_duration = total_duration / total_models if total_models > 0 else 0
        total_task_evaluations = sum(r.get('task_count', 0) for r in results)
        
        # 提取所有任务名称
        all_tasks = set()
        for result in results:
            if result.get('tasks'):
                all_tasks.update(result['tasks'])
        all_tasks = list(all_tasks)
        
        # 收集所有日志文件路径
        log_files = []
        for result in results:
            if result.get('log_file'):
                log_files.append({
                    'model_name': result['model_name'],
                    'log_file': result['log_file'],
                    'status': result['status']
                })
        
        summary = {
            "evaluation_time": datetime.now().isoformat(),
            "optimization_mode": "multi_task_single_load",
            "config": self.config,
            "log_files": log_files,  # 新增：所有日志文件信息
            "statistics": {
                "total_models": total_models,
                "successful_models": successful,
                "failed_models": failed,
                "timeout_models": timeout,
                "error_models": error,
                "success_rate": successful / total_models if total_models > 0 else 0,
                "total_duration_seconds": total_duration,
                "average_duration_per_model_seconds": avg_duration,
                "total_task_evaluations": total_task_evaluations,
                "tasks_evaluated": all_tasks,
                "efficiency_gain": f"~{len(all_tasks)}x faster than single-task mode"
            },
            "results": results
        }
        
        # 保存详细结果
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 汇总结果已保存到: {summary_file}")
        logger.info(f"📊 优化模式评测统计:")
        logger.info(f"   总模型数: {total_models}")
        logger.info(f"   成功: {successful} ({successful/total_models*100:.1f}%)")
        logger.info(f"   失败: {failed}, 超时: {timeout}, 错误: {error}")
        logger.info(f"   总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
        logger.info(f"   平均每模型: {avg_duration:.1f}秒")
        logger.info(f"   总任务评测数: {total_task_evaluations}")
        logger.info(f"   效率提升: 约{len(all_tasks)}倍于单任务模式")
        logger.info(f"📝 共生成 {len(log_files)} 个日志文件")
        
        # 保存简化的结果表格
        self.save_results_table(results, output_dir)
    
    def save_results_table(self, results: List[Dict[str, Any]], output_dir: str):
        """保存结果表格"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        table_file = Path(output_dir) / f"results_table_{timestamp}.txt"
        
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(f"优化版Ray分布式LoRA模型评测结果汇总 (多任务单次加载模式)\n")
            f.write("=" * 150 + "\n")
            f.write(f"{'模型名称':<40} {'任务数':<8} {'状态':<10} {'Ray GPU':<10} {'CUDA设备':<12} {'耗时(秒)':<10} {'日志文件':<50} {'备注'}\n")
            f.write("-" * 150 + "\n")
            
            for result in results:
                model_name = result['model_name'][:38]
                task_count = str(result.get('task_count', 0))
                status = result['status']
                ray_gpus = str(result.get('ray_gpu_ids', 'N/A'))[:8]
                cuda_devices = str(result.get('cuda_visible_devices', 'N/A'))[:10]
                duration = f"{result.get('duration', 0):.1f}"
                log_file = result.get('log_file', 'N/A')
                # 只显示日志文件名，不显示完整路径
                log_filename = Path(log_file).name if log_file != 'N/A' else 'N/A'
                log_filename = log_filename[:48]
                
                if status == 'success':
                    note = "✅ 多任务成功"
                elif status == 'failed':
                    note = "❌ " + str(result.get('error', ''))[:25]
                elif status == 'timeout':
                    note = "⏰ 超时"
                else:
                    note = "💥 异常"
                
                f.write(f"{model_name:<40} {task_count:<8} {status:<10} {ray_gpus:<10} {cuda_devices:<12} {duration:<10} {log_filename:<50} {note}\n")
            
            # 添加日志文件说明
            f.write("\n" + "=" * 150 + "\n")
            f.write("日志文件说明:\n")
            f.write("所有评测过程的详细日志都保存在 logs/ 目录下\n")
            f.write("每个模型的评测日志包含:\n")
            f.write("- 评测开始和结束时间\n")
            f.write("- 完整的命令行参数\n")
            f.write("- 评测脚本的完整输出\n")
            f.write("- 错误信息和堆栈跟踪(如果有)\n")
            f.write("- GPU和环境信息\n")
        
        logger.info(f"📋 结果表格已保存到: {table_file}")
    
    def shutdown(self):
        """关闭Ray"""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("🔚 Ray已关闭")

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="优化版Ray分布式LoRA模型评测系统 - 多任务单次加载 + 日志保存")
    
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="YAML配置文件路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="只解析配置，不执行评测")
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    logger.info("🚀 启动优化版Ray分布式LoRA模型评测系统 (多任务单次加载 + 日志保存)")
    logger.info(f"📄 配置文件: {args.config}")
    
    try:
        # 加载和验证配置
        config = ConfigManager.load_config(args.config)
        
        if not ConfigManager.validate_config(config):
            logger.error("❌ 配置文件验证失败")
            return
        
        # 如果是dry-run模式，只显示解析结果
        if args.dry_run:
            logger.info("🔍 Dry-run模式 - 解析配置:")
            model_list = ConfigManager.resolve_model_paths(config)
            eval_config = config['evaluation']
            tasks = eval_config.get('tasks', ['mmlu'])
            output_dir = eval_config.get('output_dir', './ray_results')
            log_dir = Path(output_dir) / "logs"
            
            logger.info(f"📊 将评测 {len(model_list)} 个模型:")
            for model in model_list:
                logger.info(f"  📁 {model['name']} ({model['base_model']})")
            
            logger.info(f"🎯 评测任务: {tasks}")
            logger.info(f"💾 输出目录: {output_dir}")
            logger.info(f"📝 日志目录: {log_dir}")
            logger.info(f"⚙️ 评测参数: {eval_config.get('parameters', {})}")
            
            logger.info(f"⚡ 优化模式: 每个模型一次加载，评测 {len(tasks)} 个任务")
            logger.info(f"📈 总模型数: {len(model_list)}")
            logger.info(f"🔥 效率提升: 约{len(tasks)}倍于单任务模式")
            logger.info(f"📝 每个模型将生成独立的日志文件")
            return
        
        # 创建管理器并开始评测
        manager = OptimizedRayModelEvaluationManager(config)
        
        start_time = time.time()
        results = manager.evaluate_all_models()
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in results if r['status'] == 'success')
        total_models = len(results)
        total_task_evaluations = sum(r.get('task_count', 0) for r in results)
        
        # 统计日志文件
        log_files_created = sum(1 for r in results if r.get('log_file'))
        
        logger.info(f"🎉 优化版评测完成！")
        logger.info(f"📊 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        logger.info(f"✅ 成功率: {successful}/{total_models} ({successful/total_models*100:.1f}%)")
        logger.info(f"🔥 总任务评测数: {total_task_evaluations}")
        logger.info(f"📝 生成日志文件数: {log_files_created}")
        logger.info(f"📁 结果保存在: {config['evaluation'].get('output_dir', './ray_results')}")
        logger.info(f"📝 详细日志保存在: {config['evaluation'].get('output_dir', './ray_results')}/logs/")
        
    except Exception as e:
        logger.error(f"❌ 评测过程中出现错误: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        if 'manager' in locals():
            manager.shutdown()

if __name__ == "__main__":
    main()