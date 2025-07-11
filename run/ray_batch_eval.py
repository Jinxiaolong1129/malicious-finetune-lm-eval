#!/usr/bin/env python3
# run/ray_batch_eval.py - 支持动态GPU数量和环境变量的Ray并行批量LoRA模型评测脚本

import os
import json
import yaml
import argparse
import time
import csv
import pandas as pd
import fcntl
from pathlib import Path
from typing import List, Dict, Any, Optional
import ray
from datetime import datetime
import threading


# 默认环境变量配置
DEFAULT_ENV_VARS = {
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'CUDA_LAUNCH_BLOCKING': '1',
    'HF_ALLOW_CODE_EVAL': '1',
    'TOKENIZERS_PARALLELISM': 'false'
}

def setup_environment_variables(env_vars: Dict[str, str] = None):
    """设置环境变量"""
    # 设置默认环境变量
    for key, value in DEFAULT_ENV_VARS.items():
        os.environ[key] = value
    
    # 设置用户自定义环境变量
    if env_vars:
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            print(f"🔧 设置环境变量: {key}={value}")

class ProgressTracker:
    """进度追踪器 - 使用CSV文件管理任务状态"""
    
    def __init__(self, progress_file: str = "lm_eval_experiment_progress.csv"):
        self.progress_file = Path(progress_file)
        self.lock = threading.Lock()
        self.csv_columns = [
            'lora_path', 'experiment_name', 'status', 'start_time', 'end_time', 
            'duration_minutes', 'base_model',
            'log_file', 'error_message', 'worker_pid', 
            'gpu_id', 'retry_count', 'tasks', 'created_time',
            'num_gpus_used'
        ]
        
        # 确保进度文件存在且格式正确
        self._ensure_csv_initialized()
    
    def _ensure_csv_initialized(self):
        """确保CSV文件存在且格式正确"""
        try:
            # 如果文件不存在，创建新文件
            if not self.progress_file.exists():
                self._create_new_csv()
                return
            
            # 如果文件存在但为空，重新创建
            if self.progress_file.stat().st_size == 0:
                print(f"⚠️  进度文件为空，重新初始化: {self.progress_file}")
                self._create_new_csv()
                return
            
            # 验证文件格式
            try:
                df = pd.read_csv(self.progress_file, low_memory=False)
                # 检查列是否完整
                missing_columns = set(self.csv_columns) - set(df.columns)
                if missing_columns:
                    print(f"⚠️  进度文件缺少列: {missing_columns}，重新初始化")
                    self._create_new_csv()
                    return
                print(f"📊 进度文件验证通过: {self.progress_file}")
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"⚠️  进度文件格式错误: {e}，重新初始化")
                self._create_new_csv()
                
        except Exception as e:
            print(f"❌ 检查进度文件时出错: {e}")
            self._create_new_csv()
    
    def _create_new_csv(self):
        """创建新的CSV文件"""
        try:
            # 确保目录存在
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建空的DataFrame并保存
            df = pd.DataFrame(columns=self.csv_columns)
            df.to_csv(self.progress_file, index=False)
            
            # 验证文件是否正确创建
            if self.progress_file.exists() and self.progress_file.stat().st_size > 0:
                print(f"✅ 成功初始化进度文件: {self.progress_file}")
            else:
                raise Exception("文件创建后验证失败")
                
        except Exception as e:
            print(f"❌ 创建进度文件失败: {e}")
            # 如果创建失败，尝试手动写入标题行
            try:
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    f.write(','.join(self.csv_columns) + '\n')
                print(f"✅ 手动创建进度文件成功: {self.progress_file}")
            except Exception as e2:
                print(f"❌ 手动创建进度文件也失败: {e2}")
                raise
    
    def load_progress(self) -> Dict[str, Dict[str, Any]]:
        """加载现有进度 - 增强错误处理"""
        try:
            # 再次确保文件格式正确
            self._ensure_csv_initialized()
            
            # 读取CSV文件
            df = pd.read_csv(self.progress_file, low_memory=False)
            
            if df.empty:
                print(f"📊 进度文件为空，开始新的评测")
                return {}
            
            progress = {}
            for _, row in df.iterrows():
                # 使用lora_path作为唯一标识
                lora_path = str(row['lora_path']) if pd.notna(row['lora_path']) else ''
                if lora_path:
                    progress[lora_path] = row.to_dict()
            
            print(f"📊 成功加载进度文件: {self.progress_file}")
            print(f"📈 已记录任务数: {len(progress)}")
            
            if progress:
                status_counts = df['status'].value_counts().to_dict()
                for status, count in status_counts.items():
                    print(f"  - {status}: {count}")
            
            return progress
            
        except Exception as e:
            print(f"⚠️  加载进度文件失败，将创建新文件: {e}")
            # 如果加载失败，重新创建文件
            self._create_new_csv()
            return {}
    
    def _safe_convert_value(self, key: str, value: Any) -> Any:
        """安全转换值到合适的类型"""
        if value is None or pd.isna(value):
            # 根据列类型返回合适的默认值
            if key in ['duration_minutes']:
                return 0.0
            elif key in ['retry_count', 'num_gpus_used']:
                return 0
            else:
                return ''
        try:
            if key in ['duration_minutes'] and value != '':
                return float(value)
            elif key in ['retry_count', 'num_gpus_used'] and value != '':
                return int(float(value))  # 先转float再转int，处理"1.0"这样的情况
            else:
                return str(value)
        except (ValueError, TypeError):
            if key in ['duration_minutes']:
                return 0.0
            elif key in ['retry_count', 'num_gpus_used']:
                return 0
            else:
                return str(value) if value is not None else ''
    
    def update_task_status(self, lora_path: str, **kwargs):
        """更新单个任务状态 - 增强错误处理"""
        with self.lock:
            try:
                # 确保文件存在且格式正确
                self._ensure_csv_initialized()
                
                # 读取现有数据
                try:
                    df = pd.read_csv(self.progress_file, low_memory=False)
                except (pd.errors.EmptyDataError, FileNotFoundError):
                    df = pd.DataFrame(columns=self.csv_columns)
                
                # 查找现有记录
                mask = df['lora_path'] == lora_path
                existing_idx = df.index[mask]
                
                if len(existing_idx) > 0:
                    # 更新现有记录
                    idx = existing_idx[0]
                    for key, value in kwargs.items():
                        if key in self.csv_columns:
                            converted_value = self._safe_convert_value(key, value)
                            df.at[idx, key] = converted_value
                else:
                    # 创建新记录
                    new_row = {}
                    for col in self.csv_columns:
                        if col == 'lora_path':
                            new_row[col] = str(lora_path)
                        elif col == 'created_time':
                            new_row[col] = datetime.now().isoformat()
                        elif col in kwargs:
                            new_row[col] = self._safe_convert_value(col, kwargs[col])
                        else:
                            new_row[col] = self._safe_convert_value(col, None)
                    
                    new_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_df], ignore_index=True)
                
                # 保存到文件
                df.to_csv(self.progress_file, index=False)
                
            except Exception as e:
                print(f"⚠️  更新任务状态失败 ({lora_path}): {e}")
                # 尝试重新初始化文件
                try:
                    self._create_new_csv()
                    print(f"🔄 已重新初始化进度文件，请重试")
                except Exception as e2:
                    print(f"❌ 重新初始化也失败: {e2}")

    def get_pending_experiments(self, all_experiments: List[Dict[str, Any]], 
                              force_rerun: bool = False,
                              retry_failed_only: bool = False) -> List[Dict[str, Any]]:
        """根据进度状态筛选待执行的实验"""
        if force_rerun:
            print("🔄 强制重跑模式：所有任务都将重新执行")
            return all_experiments
        
        existing_progress = self.load_progress()
        pending_experiments = []
        skipped_count = 0
        
        for exp in all_experiments:
            lora_path = str(exp['lora_path'])  # 使用lora_path作为唯一标识
            exp_name = exp['experiment_name']
            current_status = existing_progress.get(lora_path, {}).get('status', 'pending')
            
            should_run = False
            
            if retry_failed_only:
                if current_status in ['failed', 'timeout', 'error']:
                    should_run = True
                    print(f"🔄 重试失败任务: {exp_name} (上次状态: {current_status})")
                elif current_status == 'completed':
                    skipped_count += 1
                    print(f"⏭️  跳过已完成: {exp_name}")
                else:
                    should_run = True
                    print(f"▶️  执行新任务: {exp_name}")
            else:
                if current_status == 'completed':
                    skipped_count += 1
                    print(f"⏭️  跳过已完成: {exp_name}")
                elif current_status in ['failed', 'timeout', 'error', 'running']:
                    should_run = True
                    if current_status == 'running':
                        print(f"🔄 重新执行中断任务: {exp_name}")
                    else:
                        print(f"🔄 重试失败任务: {exp_name} (上次状态: {current_status})")
                else:
                    should_run = True
                    print(f"▶️  执行新任务: {exp_name}")
            
            if should_run:
                pending_experiments.append(exp)
        
        print(f"\n📊 任务筛选结果:")
        print(f"  - 总任务数: {len(all_experiments)}")
        print(f"  - 跳过已完成: {skipped_count}")
        print(f"  - 待执行任务: {len(pending_experiments)}")
        
        return pending_experiments

    def print_progress_summary(self):
        """打印进度摘要"""
        existing_progress = self.load_progress()
        if not existing_progress:
            print("📊 没有找到进度记录")
            return
        
        status_counts = {}
        for lora_path, info in existing_progress.items():
            status = info.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n📊 当前进度摘要:")
        print(f"  - 总任务数: {len(existing_progress)}")
        for status, count in status_counts.items():
            print(f"  - {status}: {count}")
            
                  
            
def create_worker_class(num_gpus: int, env_vars: Dict[str, str] = None):
    """动态创建Worker类，支持不同的GPU数量和环境变量"""
    
    @ray.remote(num_gpus=num_gpus)
    class LoRAEvaluationWorker:
        """Ray远程工作器 - 支持动态GPU数量和环境变量"""
        
        def __init__(self):
            """初始化工作器"""
            import subprocess
            import sys
            import torch
            self.subprocess = subprocess
            self.sys = sys
            self.num_gpus = num_gpus
            self.env_vars = env_vars or {}
            
            # 在Worker中设置环境变量
            for key, value in self.env_vars.items():
                os.environ[key] = str(value)
            
            self.worker_pid = os.getpid()
            if torch.cuda.is_available():
                self.gpu_id = torch.cuda.current_device()
                self.gpu_count = torch.cuda.device_count()
                
                # 如果使用多GPU，获取所有可见的GPU
                if num_gpus > 1:
                    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                    self.visible_gpus = visible_devices.split(',') if visible_devices else list(range(self.gpu_count))
                else:
                    self.visible_gpus = [self.gpu_id]
            else:
                self.gpu_id = None
                self.gpu_count = 0
                self.visible_gpus = []
                
            print(f"🔧 Worker {self.worker_pid} 初始化完成")
            print(f"🎯 分配GPU数量: {num_gpus}")
            print(f"🎯 当前GPU ID: {self.gpu_id}")
            print(f"🎯 可见GPU: {self.visible_gpus}")
            if self.env_vars:
                print(f"🔧 环境变量已设置: {len(self.env_vars)} 个")
                
        def evaluate_model(self, 
                           experiment_name: str,
                           base_model: str, 
                           lora_path: str,
                           tasks: str = "humaneval",
                           gpu_memory_utilization: float = 0.8,
                           tensor_parallel_size: Optional[int] = None) -> Dict[str, Any]:
            """评测单个LoRA模型 - 支持多GPU和环境变量"""
            start_time = time.time()
            start_time_str = datetime.now().isoformat()
            
            # 如果没有指定tensor_parallel_size，使用num_gpus
            if tensor_parallel_size is None:
                tensor_parallel_size = self.num_gpus
            
            try:
                print(f"🚀 [{experiment_name}] Worker {self.worker_pid} 开始评测")
                print(f"📁 基础模型: {base_model}")
                print(f"📁 LoRA路径: {lora_path}")
                print(f"📊 评测任务: {tasks}")
                print(f"🎯 使用GPU数量: {self.num_gpus}")
                print(f"🎯 张量并行度: {tensor_parallel_size}")
                print(f"🎯 GPU IDs: {self.visible_gpus}")
                
                # 构建输出文件路径
                lora_dir = Path(lora_path)
                log_dir = lora_dir / "log_lm_eval"
                log_dir.mkdir(parents=True, exist_ok=True)
                
                log_file = log_dir / f"{experiment_name}_{tasks}_gpu{self.num_gpus}.log"
                
                # 构建命令行参数
                cmd = [
                    self.sys.executable, "run/ray-run_evaluation.py",
                    "--base-model", base_model,
                    "--lora-path", str(lora_dir),
                    "--tasks", tasks,
                    "--output-path", str(lora_dir),
                    "--tensor-parallel-size", str(tensor_parallel_size),
                    "--gpu-memory-utilization", str(gpu_memory_utilization),
                ]

                print(f"🔄 [{experiment_name}] Worker {self.worker_pid} 执行命令: {' '.join(cmd)}")
                print(f"📝 日志文件: {log_file}")
                
                # 准备环境变量（合并默认和用户自定义）
                process_env = os.environ.copy()
                for key, value in self.env_vars.items():
                    process_env[key] = str(value)
                
                # 执行评测脚本
                with open(log_file, 'w', encoding='utf-8') as log_f:
                    log_f.write(f"=== 评测开始时间: {start_time_str} ===\n")
                    log_f.write(f"Worker PID: {self.worker_pid}\n")
                    log_f.write(f"使用GPU数量: {self.num_gpus}\n")
                    log_f.write(f"张量并行度: {tensor_parallel_size}\n")
                    log_f.write(f"GPU IDs: {self.visible_gpus}\n")
                    log_f.write(f"实验名称: {experiment_name}\n")
                    log_f.write(f"基础模型: {base_model}\n")
                    log_f.write(f"LoRA路径: {lora_path}\n")
                    log_f.write(f"评测任务: {tasks}\n")
                    log_f.write(f"执行命令: {' '.join(cmd)}\n")
                    log_f.write(f"环境变量设置: {len(self.env_vars)} 个\n")
                    for env_key, env_value in self.env_vars.items():
                        log_f.write(f"  {env_key}={env_value}\n")
                    log_f.write(f"{'='*80}\n\n")
                    log_f.flush()
                    
                    result = self.subprocess.run(
                        cmd,
                        stdout=log_f,
                        stderr=self.subprocess.STDOUT,
                        text=True,
                        timeout=3600,  # 1小时超时
                        env=process_env  # 传递环境变量
                    )
                
                # 读取日志文件内容
                try:
                    with open(log_file, 'r', encoding='utf-8') as log_f:
                        log_content = log_f.read()
                        brief_output = log_content[-1000:] if len(log_content) > 1000 else log_content
                except Exception as e:
                    brief_output = f"无法读取日志文件: {e}"
                
                end_time = time.time()
                end_time_str = datetime.now().isoformat()
                duration = end_time - start_time
                duration_minutes = duration / 60
                
                # 在日志文件末尾添加结束信息
                try:
                    with open(log_file, 'a', encoding='utf-8') as log_f:
                        log_f.write(f"\n{'='*80}\n")
                        log_f.write(f"=== 评测结束时间: {end_time_str} ===\n")
                        log_f.write(f"Worker PID: {self.worker_pid}\n")
                        log_f.write(f"返回码: {result.returncode}\n")
                        log_f.write(f"总耗时: {duration:.2f}秒 ({duration_minutes:.2f}分钟)\n")
                        log_f.write(f"状态: {'成功' if result.returncode == 0 else '失败'}\n")
                except Exception as e:
                    print(f"⚠️  [{experiment_name}] Worker {self.worker_pid} 写入日志结束信息失败: {e}")
                
                if result.returncode == 0:
                    print(f"✅ [{experiment_name}] Worker {self.worker_pid} 评测成功完成")
                    print(f"⏱️  耗时: {duration:.2f}秒 ({duration_minutes:.2f}分钟)")
                    print(f"📝 日志文件: {log_file}")
                    
                    return {
                        "experiment_name": experiment_name,
                        "lora_path": lora_path,
                        "status": "completed",
                        "start_time": start_time_str,
                        "end_time": end_time_str,
                        "duration": duration,
                        "duration_minutes": duration_minutes,
                        "log_file": str(log_file),
                        "stdout": brief_output,
                        "stderr": "",
                        "base_model": base_model,
                        "tasks": tasks,
                        "worker_pid": str(self.worker_pid),  # 转换为字符串
                        "gpu_id": str(self.gpu_id) if self.gpu_id is not None else "",  # 转换为字符串
                        "num_gpus_used": self.num_gpus,
                        "visible_gpus": self.visible_gpus,
                        "tensor_parallel_size": tensor_parallel_size,
                        "error_message": ""
                    }
                else:
                    print(f"❌ [{experiment_name}] Worker {self.worker_pid} 评测失败")
                    print(f"🔍 错误代码: {result.returncode}")
                    print(f"📝 详细日志请查看: {log_file}")
                    
                    return {
                        "experiment_name": experiment_name,
                        "lora_path": lora_path,
                        "status": "failed",
                        "start_time": start_time_str,
                        "end_time": end_time_str,
                        "duration": duration,
                        "duration_minutes": duration_minutes,
                        "error_code": result.returncode,
                        "log_file": str(log_file),
                        "stderr": "",
                        "base_model": base_model,
                        "tasks": tasks,
                        "worker_pid": str(self.worker_pid),  # 转换为字符串
                        "gpu_id": str(self.gpu_id) if self.gpu_id is not None else "",  # 转换为字符串
                        "num_gpus_used": self.num_gpus,
                        "visible_gpus": self.visible_gpus,
                        "tensor_parallel_size": tensor_parallel_size,
                        "error_message": f"Process failed with code {result.returncode}"
                    }
                    
            except self.subprocess.TimeoutExpired:
                end_time_str = datetime.now().isoformat()
                duration = time.time() - start_time
                duration_minutes = duration / 60
                
                print(f"⏰ [{experiment_name}] Worker {self.worker_pid} 评测超时")
                
                try:
                    if 'log_file' in locals():
                        with open(log_file, 'a', encoding='utf-8') as log_f:
                            log_f.write(f"\n{'='*80}\n")
                            log_f.write(f"=== 评测超时: {end_time_str} ===\n")
                            log_f.write(f"Worker PID: {self.worker_pid}\n")
                            log_f.write(f"超时时间: 3600秒 (1小时)\n")
                            log_f.write(f"实际运行时间: {duration:.2f}秒 ({duration_minutes:.2f}分钟)\n")
                except Exception:
                    pass
                
                return {
                    "experiment_name": experiment_name,
                    "lora_path": lora_path,
                    "status": "timeout",
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "duration": duration,
                    "duration_minutes": duration_minutes,
                    "error": "Evaluation timed out after 1 hour",
                    "log_file": str(log_file) if 'log_file' in locals() else None,
                    "base_model": base_model,
                    "tasks": tasks,
                    "worker_pid": str(self.worker_pid),  # 转换为字符串
                    "gpu_id": str(self.gpu_id) if self.gpu_id is not None else "",  # 转换为字符串
                    "num_gpus_used": self.num_gpus,
                    "error_message": "Evaluation timed out after 1 hour"
                }
                
            except Exception as e:
                end_time_str = datetime.now().isoformat()
                duration = time.time() - start_time
                duration_minutes = duration / 60
                
                print(f"💥 [{experiment_name}] Worker {self.worker_pid} 评测过程中出现异常: {e}")
                import traceback
                traceback.print_exc()
                
                try:
                    if 'log_file' in locals():
                        with open(log_file, 'a', encoding='utf-8') as log_f:
                            log_f.write(f"\n{'='*80}\n")
                            log_f.write(f"=== 评测异常: {end_time_str} ===\n")
                            log_f.write(f"Worker PID: {self.worker_pid}\n")
                            log_f.write(f"异常信息: {str(e)}\n")
                            log_f.write(f"异常堆栈:\n{traceback.format_exc()}\n")
                            log_f.write(f"运行时间: {duration:.2f}秒 ({duration_minutes:.2f}分钟)\n")
                except Exception:
                    pass
                
                return {
                    "experiment_name": experiment_name,
                    "lora_path": lora_path,
                    "status": "error",
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "duration": duration,
                    "duration_minutes": duration_minutes,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "log_file": str(log_file) if 'log_file' in locals() else None,
                    "base_model": base_model,
                    "tasks": tasks,
                    "worker_pid": str(self.worker_pid),  # 转换为字符串
                    "gpu_id": str(self.gpu_id) if self.gpu_id is not None else "",  # 转换为字符串
                    "num_gpus_used": self.num_gpus,
                    "error_message": str(e)
                }
    
    return LoRAEvaluationWorker


class BatchEvaluationManager:
    """批量评测管理器 - 支持动态GPU数量和环境变量"""
    
    def __init__(self, 
                 config_file: str,
                 tasks: str = "humaneval",
                 progress_file: str = "lm_eval_experiment_progress.csv",
                 num_gpus: int = 1,
                 tensor_parallel_size: Optional[int] = None,
                 env_vars: Dict[str, str] = None):
        self.config_file = config_file
        self.tasks = tasks
        self.num_gpus = num_gpus
        self.tensor_parallel_size = tensor_parallel_size or num_gpus
        self.env_vars = env_vars or {}
        
        self.progress_tracker = ProgressTracker(progress_file)
        self.all_experiments = self.load_config()
        
        self.WorkerClass = create_worker_class(num_gpus, env_vars)
        
    def load_config(self) -> List[Dict[str, Any]]:
        """加载实验配置"""
        try:
            config_path = Path(self.config_file)
            
            # 根据文件扩展名选择解析方式
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    experiments = yaml.safe_load(f)
                print(f"📁 成功加载YAML配置文件: {self.config_file}")
            else:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    experiments = json.load(f)
                print(f"📁 成功加载JSON配置文件: {self.config_file}")
            
            print(f"📊 共找到 {len(experiments)} 个实验")
            
            required_fields = ['experiment_name', 'base_model', 'lora_path']
            valid_experiments = []
            
            for i, exp in enumerate(experiments):
                missing_fields = [field for field in required_fields if field not in exp]
                if missing_fields:
                    print(f"⚠️  实验 {i} 缺少必需字段: {missing_fields}, 跳过")
                    continue
                
                if not Path(exp['lora_path']).exists():
                    print(f"⚠️  实验 '{exp['experiment_name']}' 的LoRA路径不存在: {exp['lora_path']}, 跳过")
                    continue
                
                lora_dir = Path(exp['lora_path'])
                if not lora_dir.is_dir():
                    print(f"⚠️  实验 '{exp['experiment_name']}' 的LoRA路径不是目录: {exp['lora_path']}, 跳过")
                    continue
                
                valid_experiments.append(exp)
            
            print(f"✅ 有效实验数量: {len(valid_experiments)}")
            return valid_experiments
            
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            raise
    
    def run_batch_evaluation(self, 
                        force_rerun: bool = False,
                        retry_failed_only: bool = False,
                        gpu_memory_utilization: float = 0.8) -> List[Dict[str, Any]]:
        """运行批量评测"""
        if not self.all_experiments:
            print("⚠️  没有有效的实验配置")
            return []
        
        pending_experiments = self.progress_tracker.get_pending_experiments(
            self.all_experiments, force_rerun, retry_failed_only
        )
        
        if not pending_experiments:
            print("🎉 所有任务都已完成！")
            return []
        
        # 计算实际并发数，避免创建过多等待的Actor
        import torch
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            max_concurrent = max(1, available_gpus // self.num_gpus)
            # 限制同时创建的Actor数量，避免过多等待
            batch_size = min(len(pending_experiments), max_concurrent + 2)  # +2作为缓冲
        else:
            max_concurrent = 1
            batch_size = 1
        
        print(f"\n🚀 开始批量评测")
        print(f"📊 任务: {self.tasks}")
        print(f"🔢 待执行实验数量: {len(pending_experiments)}")
        print(f"🎯 每个任务使用GPU数量: {self.num_gpus}")
        print(f"🎯 理论最大并发: {max_concurrent}")
        print(f"🎯 首批创建Actor数: {batch_size}")
        print(f"🤖 Ray将自动管理GPU资源调度")
        print(f"📊 进度文件: {self.progress_tracker.progress_file}")
        if self.env_vars:
            print(f"🔧 环境变量: {len(self.env_vars)} 个")
            for key, value in list(self.env_vars.items())[:3]:  # 只显示前3个
                print(f"   {key}={value}")
        print(f"{'='*80}")
        
        start_time = time.time()
        completed_results = []
        
        # 分批处理，避免一次创建太多等待的Actor
        experiment_batches = [pending_experiments[i:i+batch_size] for i in range(0, len(pending_experiments), batch_size)]
        
        for batch_idx, batch_experiments in enumerate(experiment_batches):
            print(f"\n🔄 处理第 {batch_idx+1}/{len(experiment_batches)} 批任务 ({len(batch_experiments)} 个)")
            
            # 为当前批次任务初始化进度状态
            for exp in batch_experiments:
                self.progress_tracker.update_task_status(
                    lora_path=exp['lora_path'],  # 使用lora_path作为唯一标识
                    experiment_name=exp['experiment_name'],
                    status='pending',
                    base_model=exp['base_model'],
                    tasks=self.tasks,
                    num_gpus_used=self.num_gpus,
                    retry_count=0
                )
            
            # 创建当前批次的Actor和任务
            task_info = {}  # future -> (lora_path, experiment_name, worker)
            
            for i, exp in enumerate(batch_experiments):
                # 更新状态为running
                self.progress_tracker.update_task_status(
                    lora_path=exp['lora_path'],  # 使用lora_path作为唯一标识
                    experiment_name=exp['experiment_name'],
                    status='running',
                    start_time=datetime.now().isoformat(),
                    num_gpus_used=self.num_gpus
                )
                
                # 创建Worker Actor
                worker = self.WorkerClass.remote()
                
                # 提交任务
                future = worker.evaluate_model.remote(
                    experiment_name=exp['experiment_name'],
                    base_model=exp['base_model'],
                    lora_path=exp['lora_path'],
                    tasks=self.tasks,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=self.tensor_parallel_size
                )
                
                # 存储任务信息
                task_info[future] = {
                    'lora_path': exp['lora_path'],  # 使用lora_path作为唯一标识
                    'experiment_name': exp['experiment_name'],
                    'worker': worker
                }
                
                current_task_num = batch_idx * batch_size + i + 1
                print(f"📤 已提交任务 {current_task_num}/{len(pending_experiments)}: {exp['experiment_name']} (需要 {self.num_gpus} GPU)")
            
            print(f"\n⏳ 等待当前批次 {len(task_info)} 个任务完成...")
            if len(task_info) > max_concurrent:
                print(f"💡 注意: 当前批次任务数({len(task_info)})超过并发数({max_concurrent})，部分任务将排队等待")
            
            # 等待当前批次所有任务完成
            remaining_futures = list(task_info.keys())
            
            while remaining_futures:
                ready_futures, remaining_futures = ray.wait(
                    remaining_futures, 
                    num_returns=1, 
                    timeout=30.0
                )
                
                for ready_future in ready_futures:
                    task_lora_path = None
                    task_name = None
                    worker = None
                    
                    if ready_future in task_info:
                        task_lora_path = task_info[ready_future]['lora_path']
                        task_name = task_info[ready_future]['experiment_name']
                        worker = task_info[ready_future]['worker']
                        del task_info[ready_future]
                    
                    try:
                        result = ray.get(ready_future)
                        completed_results.append(result)
                        
                        # 更新进度文件 - 使用lora_path作为唯一标识
                        self.progress_tracker.update_task_status(
                            lora_path=result['lora_path'],
                            experiment_name=result['experiment_name'],
                            status=result['status'],
                            end_time=result.get('end_time', datetime.now().isoformat()),
                            duration_minutes=result.get('duration_minutes', 0),
                            log_file=result.get('log_file', ''),
                            error_message=result.get('error_message', ''),
                            worker_pid=result.get('worker_pid'),
                            gpu_id=result.get('gpu_id'),
                            num_gpus_used=result.get('num_gpus_used', self.num_gpus)
                        )
                        
                        # 打印完成信息
                        status_emoji = "✅" if result['status'] == 'completed' else "❌"
                        duration_info = f" [{result.get('duration_minutes', 0):.1f}分钟]"
                        gpu_info = f" [使用{result.get('num_gpus_used', self.num_gpus)}GPU]"
                        print(f"{status_emoji} {result['experiment_name']}{duration_info}{gpu_info}")
                        
                        if worker:
                            try:
                                ray.kill(worker)
                            except Exception as e:
                                print(f"⚠️  清理Worker失败: {e}")
                        
                    except Exception as e:
                        print(f"❌ 获取任务结果失败 ({task_name}): {e}")
                        if task_lora_path:
                            self.progress_tracker.update_task_status(
                                lora_path=task_lora_path,
                                experiment_name=task_name or 'unknown',
                                status='error',
                                end_time=datetime.now().isoformat(),
                                error_message=f"Failed to get result: {str(e)}",
                                num_gpus_used=self.num_gpus
                            )
                        
                        if worker:
                            try:
                                ray.kill(worker)
                            except Exception:
                                pass
                
                if remaining_futures:
                    completed_count = len(completed_results)
                    total_count = len(pending_experiments)
                    batch_remaining = len(remaining_futures)
                    progress_percent = completed_count / total_count * 100
                    print(f"📈 总进度: {completed_count}/{total_count} ({progress_percent:.1f}%) | 当前批次剩余: {batch_remaining}")
            
            # 清理当前批次剩余的worker引用
            for future, info in task_info.items():
                try:
                    ray.kill(info['worker'])
                except Exception:
                    pass
            
            print(f"✅ 第 {batch_idx+1} 批任务完成")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 统计结果
        successful = [r for r in completed_results if r['status'] == 'completed']
        failed = [r for r in completed_results if r['status'] != 'completed']
        
        print(f"\n{'='*80}")
        print(f"📊 批量评测完成!")
        print(f"⏱️  总耗时: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟)")
        print(f"🎯 每任务GPU数: {self.num_gpus}")
        print(f"🎯 张量并行度: {self.tensor_parallel_size}")
        print(f"✅ 成功: {len(successful)}")
        print(f"❌ 失败: {len(failed)}")
        if completed_results:
            print(f"📈 成功率: {len(successful)/len(completed_results)*100:.1f}%")
        print(f"📊 进度文件: {self.progress_tracker.progress_file}")

        # 显示成功任务的执行时间
        if successful:
            print(f"\n🏆 成功任务执行时间:")
            successful_sorted = sorted(successful, key=lambda x: x.get('duration_minutes', 0))
            
            for i, result in enumerate(successful_sorted[:10], 1):
                worker_info = f"(Worker {result.get('worker_pid', 'N/A')}, {result.get('num_gpus_used', self.num_gpus)}GPU)"
                duration_info = f"[{result.get('duration_minutes', 0):.1f}min]"
                print(f"  {i:2d}. {result['experiment_name']:<25}: {duration_info} {worker_info}")
        
        # 显示失败任务
        if failed:
            print(f"\n❌ 失败任务详情:")
            for result in failed:
                worker_info = f"Worker {result.get('worker_pid', 'N/A')}, {result.get('num_gpus_used', self.num_gpus)}GPU"
                duration_info = f"[{result.get('duration_minutes', 0):.1f}min]"
                error_msg = result.get('error_message', result.get('error', 'Unknown error'))
                print(f"  - {result['experiment_name']}: {result['status']} - {error_msg} {duration_info} ({worker_info})")
        
        # 保存详细结果到汇总目录
        summary_dir = Path("./batch_evaluation_summaries")
        summary_dir.mkdir(parents=True, exist_ok=True)
        self.save_batch_results(completed_results, total_duration, summary_dir)
        
        return completed_results
    
    
    def save_batch_results(self, results: List[Dict[str, Any]], total_duration: float, summary_dir: Path):
        """保存批量评测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"batch_evaluation_summary_{timestamp}.json"
        
        summary = {
            "evaluation_time": datetime.now().isoformat(),
            "config_file": str(self.config_file),
            "tasks": self.tasks,
            "num_gpus_per_task": self.num_gpus,
            "tensor_parallel_size": self.tensor_parallel_size,
            "environment_variables": self.env_vars,
            "total_experiments": len(self.all_experiments),
            "pending_experiments": len(results) if results else 0,
            "total_duration_seconds": total_duration,
            "total_duration_minutes": total_duration / 60,
            "output_mode": "lora_directories",
            "scheduling_mode": "ray_auto_scheduling",
            "progress_file": str(self.progress_tracker.progress_file),
            "results_summary": {
                "successful": len([r for r in results if r['status'] == 'completed']),
                "failed": len([r for r in results if r['status'] != 'completed']),
                "success_rate": len([r for r in results if r['status'] == 'completed']) / len(results) * 100 if results else 0
            },
            "detailed_results": results
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📄 详细结果已保存到: {summary_file}")
            
        except Exception as e:
            print(f"⚠️  保存结果摘要失败: {e}")

def load_config_from_yaml(config_file: str) -> Dict[str, Any]:
    """从YAML配置文件加载所有配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"📁 成功加载YAML配置: {config_file}")
        return config
    except Exception as e:
        print(f"❌ 加载YAML配置失败: {e}")
        raise

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="支持动态GPU数量和环境变量的Ray并行批量LoRA模型评测脚本")
    
    # 配置文件参数 - 修改为可选
    parser.add_argument("--config", type=str, default=None,
                        help="实验配置JSON/YAML文件路径（可通过yaml-config中的config字段指定）")
    parser.add_argument("--yaml-config", type=str, required=True,
                        help="YAML格式的完整配置文件（包含所有参数和环境变量）")
    
    # 评测参数
    parser.add_argument("--tasks", type=str, default=None,
                        help="评测任务（将覆盖YAML配置）")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="每个任务使用的GPU数量（将覆盖YAML配置）")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="张量并行大小（将覆盖YAML配置）")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="GPU内存利用率（将覆盖YAML配置）")
    
    # Ray配置
    parser.add_argument("--ray-address", type=str, default=None,
                        help="Ray集群地址（将覆盖YAML配置）")
    
    # 进度管理
    parser.add_argument("--progress-file", type=str, default=None,
                        help="进度追踪文件路径（将覆盖YAML配置）")
    parser.add_argument("--force-rerun", action="store_true",
                        help="强制重跑所有任务（忽略已完成状态）")
    parser.add_argument("--retry-failed-only", action="store_true",
                        help="只重新运行失败的任务")
    parser.add_argument("--show-progress", action="store_true",
                        help="只显示当前进度，不执行评测")
    
    # 环境变量
    parser.add_argument("--env", action="append", nargs=2, metavar=("KEY", "VALUE"),
                        help="设置环境变量，格式: --env KEY VALUE (可多次使用，将与YAML配置合并)")
    
    return parser

def merge_configs(yaml_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """合并YAML配置和命令行参数，命令行参数优先级更高"""
    merged_config = yaml_config.copy()
    
    # 映射命令行参数到配置键
    arg_to_config_mapping = {
        'config': 'config',
        'tasks': 'tasks',
        'num_gpus': 'num_gpus',
        'tensor_parallel_size': 'tensor_parallel_size',
        'gpu_memory_utilization': 'gpu_memory_utilization',
        'ray_address': 'ray_address',
        'progress_file': 'progress_file'
    }
    
    # 从命令行参数覆盖YAML配置
    for arg_name, config_key in arg_to_config_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            merged_config[config_key] = arg_value
            print(f"🔧 命令行参数覆盖YAML配置: {config_key}={arg_value}")
    
    # 处理环境变量合并
    env_vars = merged_config.get('environment_variables', {})
    if args.env:
        print(f"🔧 合并命令行环境变量:")
        for key, value in args.env:
            env_vars[key] = value
            print(f"   {key}={value}")
        merged_config['environment_variables'] = env_vars
    
    return merged_config

def validate_gpu_config(num_gpus: int, tensor_parallel_size: Optional[int] = None) -> tuple:
    """验证GPU配置的合理性"""
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA不可用，将在CPU模式下运行")
        return 0, 1
    
    available_gpus = torch.cuda.device_count()
    print(f"🎯 系统可用GPU数量: {available_gpus}")
    
    if num_gpus > available_gpus:
        print(f"⚠️  警告: 请求的GPU数量 ({num_gpus}) 超过可用数量 ({available_gpus})")
        print(f"🔧 自动调整为最大可用数量: {available_gpus}")
        num_gpus = available_gpus
    
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
    
    if tensor_parallel_size > num_gpus:
        print(f"⚠️  警告: 张量并行大小 ({tensor_parallel_size}) 大于GPU数量 ({num_gpus})")
        print(f"🔧 自动调整张量并行大小为: {num_gpus}")
        tensor_parallel_size = num_gpus
    
    return num_gpus, tensor_parallel_size

def estimate_concurrent_tasks(num_gpus_per_task: int, available_gpus: int) -> int:
    """估算最大并发任务数"""
    if num_gpus_per_task == 0:
        return 1  # CPU模式
    
    max_concurrent = available_gpus // num_gpus_per_task
    return max(1, max_concurrent)

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 加载YAML配置文件
    if not args.yaml_config:
        print("❌ 错误: 必须提供 --yaml-config 参数")
        return
    
    yaml_config = load_config_from_yaml(args.yaml_config)
    
    # 合并配置
    config = merge_configs(yaml_config, args)
    
    # 验证必需的配置
    if 'config' not in config:
        print("❌ 错误: YAML配置文件中必须包含 'config' 字段指定实验配置文件路径")
        return
    
    # 设置默认值
    tasks = config.get('tasks', 'humaneval')
    num_gpus = config.get('num_gpus', 1)
    tensor_parallel_size = config.get('tensor_parallel_size', None)
    gpu_memory_utilization = config.get('gpu_memory_utilization', 0.8)
    progress_file = config.get('progress_file', 'lm_eval_experiment_progress.csv')
    ray_address = config.get('ray_address', None)
    env_vars = config.get('environment_variables', {})
    
    print("🚀 支持动态GPU数量和环境变量的Ray并行批量LoRA模型评测系统")
    print(f"📁 YAML配置文件: {args.yaml_config}")
    print(f"📁 实验配置文件: {config['config']}")
    print(f"📊 评测任务: {tasks}")
    print(f"🎯 每任务GPU数量: {num_gpus}")
    print(f"🎯 张量并行大小: {tensor_parallel_size or num_gpus}")
    print(f"💾 GPU内存利用率: {gpu_memory_utilization}")
    print(f"📊 进度文件: {progress_file}")
    
    # 设置环境变量
    setup_environment_variables(env_vars)
    
    # 验证GPU配置
    validated_num_gpus, validated_tensor_parallel_size = validate_gpu_config(
        num_gpus, tensor_parallel_size
    )
    
    if validated_num_gpus != num_gpus:
        num_gpus = validated_num_gpus
    if validated_tensor_parallel_size != (tensor_parallel_size or num_gpus):
        tensor_parallel_size = validated_tensor_parallel_size
    
    # 估算并发任务数
    if validated_num_gpus > 0:
        import torch
        available_gpus = torch.cuda.device_count()
        max_concurrent = estimate_concurrent_tasks(num_gpus, available_gpus)
        print(f"📈 预估最大并发任务数: {max_concurrent} (基于 {available_gpus} 个GPU)")
        
        if max_concurrent == 1 and num_gpus > 1:
            print(f"💡 提示: 由于每个任务需要 {num_gpus} GPU，任务将串行执行")
        elif max_concurrent > 1:
            total_gpu_usage = max_concurrent * num_gpus
            print(f"💡 提示: 最多 {max_concurrent} 个任务并行，总计使用 {total_gpu_usage} GPU")
    
    if args.force_rerun:
        print(f"🔄 运行模式: 强制重跑所有任务")
    elif args.retry_failed_only:
        print(f"🔄 运行模式: 只重跑失败任务")
    else:
        print(f"🔄 运行模式: 智能续传 (跳过已完成)")
    
    print(f"🤖 调度模式: Ray自动资源管理")
    print(f"🔧 特性: 支持动态GPU配置和环境变量")
    
    # 如果只是查看进度，不需要初始化Ray
    if args.show_progress:
        print(f"\n📊 显示当前进度:")
        progress_tracker = ProgressTracker(progress_file)
        progress_tracker.print_progress_summary()
        return
    
    # 初始化Ray
    if ray_address:
        print(f"🌐 连接到Ray集群: {ray_address}")
        ray.init(address=ray_address)
    else:
        print(f"🖥️  启动本地Ray集群")
        ray.init()
    
    try:
        # 创建批量评测管理器
        manager = BatchEvaluationManager(
            config_file=config['config'],
            tasks=tasks,
            progress_file=progress_file,
            num_gpus=num_gpus,
            tensor_parallel_size=tensor_parallel_size,
            env_vars=env_vars
        )
        
        # 运行批量评测
        results = manager.run_batch_evaluation(
            force_rerun=args.force_rerun,
            retry_failed_only=args.retry_failed_only,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
        print(f"\n🎉 所有任务处理完成!")
        print(f"📊 进度文件: {progress_file}")
        print(f"📄 汇总文件: ./batch_evaluation_summaries/")
        print(f"💡 提示: 使用 --show-progress 可以随时查看进度")
        print(f"💡 提示: 使用 --retry-failed-only 可以只重跑失败的任务")
        print(f"💡 提示: 使用 --num-gpus N 可以指定每个任务使用的GPU数量")
        print(f"💡 提示: 通过修改YAML配置文件可以调整所有参数")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断评测")
        print(f"📊 当前进度已保存到: {progress_file}")
        print(f"💡 可以使用相同命令重新启动以继续未完成的任务")
    except Exception as e:
        print(f"💥 评测过程出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭Ray
        try:
            ray.shutdown()
            print(f"🔄 Ray集群已关闭")
        except:
            pass

if __name__ == "__main__":
    main()