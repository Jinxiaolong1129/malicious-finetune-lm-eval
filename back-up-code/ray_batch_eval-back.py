#!/usr/bin/env python3
# ray_batch_eval_simplified.py - 简化版Ray并行批量LoRA模型评测脚本（只追踪成功运行状态）

import os
import json
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

# 设置环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class ProgressTracker:
    """进度追踪器 - 使用CSV文件管理任务状态"""
    
    def __init__(self, progress_file: str = "lm_eval_experiment_progress.csv"):
        self.progress_file = Path(progress_file)
        self.lock = threading.Lock()
        self.csv_columns = [
            'experiment_name', 'status', 'start_time', 'end_time', 
            'duration_minutes', 'base_model', 'lora_path', 
            'log_file', 'error_message', 'worker_pid', 
            'gpu_id', 'retry_count', 'tasks', 'created_time'
        ]
        
        if not self.progress_file.exists():
            self._initialize_csv()
    
    def _initialize_csv(self):
        """初始化CSV文件"""
        try:
            with open(self.progress_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()
            print(f"📊 初始化进度文件: {self.progress_file}")
        except Exception as e:
            print(f"❌ 初始化进度文件失败: {e}")
            raise
    
    def load_progress(self) -> Dict[str, Dict[str, Any]]:
        """加载现有进度"""
        if not self.progress_file.exists():
            return {}
        
        try:
            df = pd.read_csv(self.progress_file)
            if df.empty:
                return {}
            
            progress = {}
            for _, row in df.iterrows():
                progress[row['experiment_name']] = row.to_dict()
            
            print(f"📊 加载进度文件: {self.progress_file}")
            print(f"📈 已记录任务数: {len(progress)}")
            
            status_counts = df['status'].value_counts().to_dict()
            for status, count in status_counts.items():
                print(f"  - {status}: {count}")
            
            return progress
            
        except Exception as e:
            print(f"⚠️  加载进度文件失败: {e}")
            return {}
    
    def update_task_status(self, experiment_name: str, **kwargs):
        """更新单个任务状态"""
        with self.lock:
            try:
                if self.progress_file.exists():
                    df = pd.read_csv(self.progress_file)
                else:
                    df = pd.DataFrame(columns=self.csv_columns)
                
                mask = df['experiment_name'] == experiment_name
                existing_idx = df.index[mask]
                
                if len(existing_idx) > 0:
                    # 更新现有记录
                    idx = existing_idx[0]
                    for key, value in kwargs.items():
                        if key in self.csv_columns:
                            # 确保数据类型一致性
                            if value is not None:
                                df.at[idx, key] = str(value)
                            else:
                                df.at[idx, key] = ''
                else:
                    # 创建新记录
                    new_row = {col: '' for col in self.csv_columns}
                    new_row['experiment_name'] = experiment_name
                    new_row['created_time'] = datetime.now().isoformat()
                    
                    # 确保所有值都是字符串类型
                    for key, value in kwargs.items():
                        if key in self.csv_columns:
                            new_row[key] = str(value) if value is not None else ''
                    
                    new_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_df], ignore_index=True)
                
                # 保存到文件
                df.to_csv(self.progress_file, index=False)
                
            except Exception as e:
                print(f"⚠️  更新任务状态失败 ({experiment_name}): {e}")
    
    def get_task_status(self, experiment_name: str) -> Optional[str]:
        """获取任务状态"""
        try:
            if not self.progress_file.exists():
                return None
            
            df = pd.read_csv(self.progress_file)
            mask = df['experiment_name'] == experiment_name
            if mask.any():
                return df.loc[mask, 'status'].iloc[0]
            return None
            
        except Exception as e:
            print(f"⚠️  获取任务状态失败 ({experiment_name}): {e}")
            return None
    
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
            exp_name = exp['experiment_name']
            current_status = existing_progress.get(exp_name, {}).get('status', 'pending')
            
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
        try:
            if not self.progress_file.exists():
                print("📊 暂无进度记录")
                return
            
            df = pd.read_csv(self.progress_file)
            if df.empty:
                print("📊 进度文件为空")
                return
            
            print(f"\n📊 当前进度摘要:")
            print(f"   进度文件: {self.progress_file}")
            
            status_counts = df['status'].value_counts()
            total = len(df)
            
            for status, count in status_counts.items():
                percentage = count / total * 100
                print(f"   {status}: {count} ({percentage:.1f}%)")
            
            # 显示最近的任务信息
            if 'created_time' in df.columns:
                recent_tasks = df.nlargest(5, 'created_time')
                print(f"\n📝 最近5个任务:")
                for _, row in recent_tasks.iterrows():
                    print(f"   {row['experiment_name']}: {row['status']}")
                    
        except Exception as e:
            print(f"⚠️  打印进度摘要失败: {e}")


@ray.remote(num_gpus=1)
class LoRAEvaluationWorker:
    """Ray远程工作器 - 每个工作器使用1个GPU"""
    
    def __init__(self):
        """初始化工作器"""
        import subprocess
        import sys
        import torch
        self.subprocess = subprocess
        self.sys = sys
        
        self.worker_pid = os.getpid()
        if torch.cuda.is_available():
            self.gpu_id = torch.cuda.current_device()
            self.gpu_count = torch.cuda.device_count()
        else:
            self.gpu_id = None
            self.gpu_count = 0
            
        print(f"🔧 Worker {self.worker_pid} 初始化完成，GPU: {self.gpu_id}/{self.gpu_count}")
            
    def evaluate_model(self, 
                       experiment_name: str,
                       base_model: str, 
                       lora_path: str,
                       tasks: str = "humaneval",
                       gpu_memory_utilization: float = 0.8) -> Dict[str, Any]:
        """评测单个LoRA模型 - 简化版，只追踪运行状态"""
        start_time = time.time()
        start_time_str = datetime.now().isoformat()
        
        try:
            print(f"🚀 [{experiment_name}] Worker {self.worker_pid} 开始评测")
            print(f"📁 基础模型: {base_model}")
            print(f"📁 LoRA路径: {lora_path}")
            print(f"📊 评测任务: {tasks}")
            print(f"🎯 使用GPU: {self.gpu_id}")
            
            # 构建输出文件路径 - 保存到LoRA目录
            lora_dir = Path(lora_path)
            log_dir = lora_dir / "log_lm_eval"
            
            print(f"📝 日志保存到: {log_dir}")
            
            # 确保输出目录和日志目录存在
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{experiment_name}_{tasks}.log"
            
            # 构建命令行参数
            cmd = [
                self.sys.executable, "ray-run_evaluation.py",
                "--base-model", base_model,
                "--lora-path", str(lora_dir),
                "--tasks", tasks,
                "--output-path", str(lora_dir),
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", str(gpu_memory_utilization),
            ]
            
            print(f"🔄 [{experiment_name}] Worker {self.worker_pid} 执行命令: {' '.join(cmd)}")
            print(f"📝 日志文件: {log_file}")
            
            # 执行评测脚本
            with open(log_file, 'w', encoding='utf-8') as log_f:
                log_f.write(f"=== 评测开始时间: {start_time_str} ===\n")
                log_f.write(f"Worker PID: {self.worker_pid}\n")
                log_f.write(f"GPU ID: {self.gpu_id}\n")
                log_f.write(f"实验名称: {experiment_name}\n")
                log_f.write(f"基础模型: {base_model}\n")
                log_f.write(f"LoRA路径: {lora_path}\n")
                log_f.write(f"评测任务: {tasks}\n")
                log_f.write(f"执行命令: {' '.join(cmd)}\n")
                log_f.write(f"{'='*80}\n\n")
                log_f.flush()
                
                result = self.subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=self.subprocess.STDOUT,
                    text=True,
                    timeout=3600  # 1小时超时
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
                    "status": "completed",
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "duration": duration,
                    "duration_minutes": duration_minutes,
                    "log_file": str(log_file),
                    "stdout": brief_output,
                    "stderr": "",
                    "base_model": base_model,
                    "lora_path": lora_path,
                    "tasks": tasks,
                    "worker_pid": self.worker_pid,
                    "gpu_id": self.gpu_id,
                    "error_message": ""
                }
            else:
                print(f"❌ [{experiment_name}] Worker {self.worker_pid} 评测失败")
                print(f"🔍 错误代码: {result.returncode}")
                print(f"📝 详细日志请查看: {log_file}")
                
                return {
                    "experiment_name": experiment_name,
                    "status": "failed",
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "duration": duration,
                    "duration_minutes": duration_minutes,
                    "error_code": result.returncode,
                    "log_file": str(log_file),
                    "stdout": brief_output,
                    "stderr": "",
                    "base_model": base_model,
                    "lora_path": lora_path,
                    "tasks": tasks,
                    "worker_pid": self.worker_pid,
                    "gpu_id": self.gpu_id,
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
                "status": "timeout",
                "start_time": start_time_str,
                "end_time": end_time_str,
                "duration": duration,
                "duration_minutes": duration_minutes,
                "error": "Evaluation timed out after 1 hour",
                "log_file": str(log_file) if 'log_file' in locals() else None,
                "base_model": base_model,
                "lora_path": lora_path,
                "tasks": tasks,
                "worker_pid": self.worker_pid,
                "gpu_id": self.gpu_id,
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
                "status": "error",
                "start_time": start_time_str,
                "end_time": end_time_str,
                "duration": duration,
                "duration_minutes": duration_minutes,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "log_file": str(log_file) if 'log_file' in locals() else None,
                "base_model": base_model,
                "lora_path": lora_path,
                "tasks": tasks,
                "worker_pid": self.worker_pid,
                "gpu_id": self.gpu_id,
                "error_message": str(e)
            }


class BatchEvaluationManager:
    """批量评测管理器 - 简化版"""
    
    def __init__(self, 
                 config_file: str,
                 tasks: str = "humaneval",
                 progress_file: str = "lm_eval_experiment_progress.csv"):
        self.config_file = config_file
        self.tasks = tasks
        
        self.progress_tracker = ProgressTracker(progress_file)
        self.all_experiments = self.load_config()
        
    def load_config(self) -> List[Dict[str, Any]]:
        """加载实验配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                experiments = json.load(f)
            
            print(f"📁 成功加载配置文件: {self.config_file}")
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
                
                try:
                    test_file = lora_dir / ".write_test"
                    test_file.touch()
                    test_file.unlink()
                except Exception:
                    print(f"⚠️  实验 '{exp['experiment_name']}' 的LoRA目录无写入权限: {exp['lora_path']}, 跳过")
                    continue
                
                valid_experiments.append(exp)
            
            print(f"✅ 有效实验数量: {len(valid_experiments)}")
            return valid_experiments
            
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            raise
    
    def run_batch_evaluation(self, 
                           force_rerun: bool = False,
                           retry_failed_only: bool = False) -> List[Dict[str, Any]]:
        """运行批量评测"""
        if not self.all_experiments:
            print("⚠️  没有有效的实验配置")
            return []
        
        self.progress_tracker.print_progress_summary()
        
        pending_experiments = self.progress_tracker.get_pending_experiments(
            self.all_experiments, force_rerun, retry_failed_only
        )
        
        if not pending_experiments:
            print("🎉 所有任务都已完成！")
            return []
        
        print(f"\n🚀 开始批量评测")
        print(f"📊 任务: {self.tasks}")
        print(f"🔢 待执行实验数量: {len(pending_experiments)}")
        print(f"🎯 每个任务使用 1 GPU")
        print(f"🤖 Ray将自动管理GPU资源调度")
        print(f"💾 输出模式: 保存到各自的LoRA目录")
        print(f"📊 进度文件: {self.progress_tracker.progress_file}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # 为所有待执行任务初始化进度状态
        for exp in pending_experiments:
            self.progress_tracker.update_task_status(
                experiment_name=exp['experiment_name'],
                status='pending',
                base_model=exp['base_model'],
                lora_path=exp['lora_path'],
                tasks=self.tasks,
                retry_count=0
            )
        
        # 创建持久的Actor引用
        print(f"🔧 创建 {len(pending_experiments)} 个Worker Actor...")
        workers = []
        futures = []
        task_names = []
        
        for i, exp in enumerate(pending_experiments):
            # 更新状态为running
            self.progress_tracker.update_task_status(
                experiment_name=exp['experiment_name'],
                status='running',
                start_time=datetime.now().isoformat()
            )
            
            # 创建持久的Worker Actor引用
            worker = LoRAEvaluationWorker.remote()
            workers.append(worker)  # 保存引用防止被垃圾回收
            
            # 提交任务
            future = worker.evaluate_model.remote(
                experiment_name=exp['experiment_name'],
                base_model=exp['base_model'],
                lora_path=exp['lora_path'],
                tasks=self.tasks
            )
            
            futures.append(future)
            task_names.append(exp['experiment_name'])
            print(f"📤 已提交任务 {i+1}/{len(pending_experiments)}: {exp['experiment_name']}")
        
        print(f"\n⏳ 等待 {len(futures)} 个任务完成...")
        print(f"🤖 Ray会根据可用GPU自动调度任务执行")
        print(f"📊 可以随时查看进度文件: {self.progress_tracker.progress_file}")
        
        # 使用ray.get等待所有任务完成，同时支持实时进度更新
        completed_results = []
        remaining_futures = list(zip(futures, task_names, workers))
        
        while remaining_futures:
            # 等待至少一个任务完成
            ready_futures, remaining_future_pairs = ray.wait(
                [f for f, _, _ in remaining_futures], 
                num_returns=1, 
                timeout=30.0  # 30秒超时，用于定期检查
            )
            
            # 处理已完成的任务
            for ready_future in ready_futures:
                # 找到对应的任务名称和worker
                task_name = None
                worker = None
                for i, (f, name, w) in enumerate(remaining_future_pairs):
                    if f == ready_future:
                        task_name = name
                        worker = w
                        remaining_future_pairs.pop(i)
                        break
                
                # 获取结果并更新进度
                try:
                    result = ray.get(ready_future)
                    completed_results.append(result)
                    
                    # 更新进度文件
                    self.progress_tracker.update_task_status(
                        experiment_name=result['experiment_name'],
                        status=result['status'],
                        end_time=result.get('end_time', datetime.now().isoformat()),
                        duration_minutes=result.get('duration_minutes', 0),
                        log_file=result.get('log_file', ''),
                        error_message=result.get('error_message', ''),
                        worker_pid=result.get('worker_pid'),
                        gpu_id=result.get('gpu_id')
                    )
                    
                    # 打印完成信息
                    status_emoji = "✅" if result['status'] == 'completed' else "❌"
                    duration_info = f" [{result.get('duration_minutes', 0):.1f}分钟]"
                    print(f"{status_emoji} {result['experiment_name']}{duration_info}")
                    
                    # 任务完成后清理worker
                    if worker:
                        try:
                            ray.kill(worker)
                        except Exception as e:
                            print(f"⚠️  清理Worker失败: {e}")
                    
                except Exception as e:
                    print(f"❌ 获取任务结果失败 ({task_name}): {e}")
                    # 即使获取结果失败，也要更新状态
                    if task_name:
                        self.progress_tracker.update_task_status(
                            experiment_name=task_name,
                            status='error',
                            end_time=datetime.now().isoformat(),
                            error_message=f"Failed to get result: {str(e)}"
                        )
                    
                    # 清理worker
                    if worker:
                        try:
                            ray.kill(worker)
                        except Exception:
                            pass
            
            # 更新剩余任务列表
            remaining_futures = remaining_future_pairs
            
            # 如果还有任务在运行，打印进度信息
            if remaining_futures:
                completed_count = len(completed_results)
                total_count = len(pending_experiments)
                progress_percent = completed_count / total_count * 100
                print(f"📈 进度: {completed_count}/{total_count} ({progress_percent:.1f}%) - 剩余 {len(remaining_futures)} 个任务运行中...")
        
        # 清理剩余的worker引用
        for worker in workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 确保所有结果都已收集
        if len(completed_results) < len(pending_experiments):
            print(f"⚠️  预期 {len(pending_experiments)} 个结果，实际收集到 {len(completed_results)} 个")
        
        # 统计结果
        successful = [r for r in completed_results if r['status'] == 'completed']
        failed = [r for r in completed_results if r['status'] != 'completed']
        
        print(f"\n{'='*80}")
        print(f"📊 批量评测完成!")
        print(f"⏱️  总耗时: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟)")
        print(f"✅ 成功: {len(successful)}")
        print(f"❌ 失败: {len(failed)}")
        if completed_results:
            print(f"📈 成功率: {len(successful)/len(completed_results)*100:.1f}%")
        print(f"📊 进度文件: {self.progress_tracker.progress_file}")
        
        # 显示成功任务的执行时间
        if successful:
            print(f"\n🏆 成功任务执行时间:")
            successful_sorted = sorted(successful, key=lambda x: x.get('duration_minutes', 0))
            
            for i, result in enumerate(successful_sorted[:10], 1):  # 显示前10名
                worker_info = f"(Worker {result.get('worker_pid', 'N/A')}, GPU {result.get('gpu_id', 'N/A')})"
                duration_info = f"[{result.get('duration_minutes', 0):.1f}min]"
                print(f"  {i:2d}. {result['experiment_name']:<25}: {duration_info} {worker_info}")
        
        # 显示失败任务
        if failed:
            print(f"\n❌ 失败任务详情:")
            for result in failed:
                worker_info = f"Worker {result.get('worker_pid', 'N/A')}, GPU {result.get('gpu_id', 'N/A')}"
                duration_info = f"[{result.get('duration_minutes', 0):.1f}min]"
                error_msg = result.get('error_message', result.get('error', 'Unknown error'))
                print(f"  - {result['experiment_name']}: {result['status']} - {error_msg} {duration_info} ({worker_info})")
        
        # 保存详细结果到汇总目录
        summary_dir = Path("./batch_evaluation_summaries")
        summary_dir.mkdir(parents=True, exist_ok=True)
        self.save_batch_results(completed_results, total_duration, summary_dir)
        
        # 最后再次显示进度摘要
        print(f"\n📊 最终进度摘要:")
        self.progress_tracker.print_progress_summary()
        
        return completed_results
    
    def save_batch_results(self, results: List[Dict[str, Any]], total_duration: float, summary_dir: Path):
        """保存批量评测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"batch_evaluation_summary_{timestamp}.json"
        
        summary = {
            "evaluation_time": datetime.now().isoformat(),
            "config_file": str(self.config_file),
            "tasks": self.tasks,
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

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="简化版Ray并行批量LoRA模型评测脚本")
    
    parser.add_argument("--config", type=str, required=True,
                        help="实验配置JSON文件路径")
    parser.add_argument("--tasks", type=str, default="humaneval",
                        help="评测任务")
    parser.add_argument("--ray-address", type=str, default=None,
                        help="Ray集群地址 (None表示本地模式)")
    parser.add_argument("--progress-file", type=str, default="lm_eval_experiment_progress.csv",
                        help="进度追踪文件路径")
    parser.add_argument("--force-rerun", action="store_true",
                        help="强制重跑所有任务（忽略已完成状态）")
    parser.add_argument("--retry-failed-only", action="store_true",
                        help="只重新运行失败的任务")
    parser.add_argument("--show-progress", action="store_true",
                        help="只显示当前进度，不执行评测")
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 简化版Ray并行批量LoRA模型评测系统")
    print(f"📁 配置文件: {args.config}")
    print(f"📊 评测任务: {args.tasks}")
    print(f"📊 进度文件: {args.progress_file}")
    print(f"💾 输出模式: 保存到各自的LoRA目录 (默认)")
    
    if args.force_rerun:
        print(f"🔄 运行模式: 强制重跑所有任务")
    elif args.retry_failed_only:
        print(f"🔄 运行模式: 只重跑失败任务")
    else:
        print(f"🔄 运行模式: 智能续传 (跳过已完成)")
    
    print(f"🤖 调度模式: Ray自动资源管理")
    print(f"🎯 每任务GPU数: 1")
    print(f"🔧 特性: 简化版本，只追踪运行状态")
    
    # 如果只是查看进度，不需要初始化Ray
    if args.show_progress:
        print(f"\n📊 显示当前进度:")
        progress_tracker = ProgressTracker(args.progress_file)
        progress_tracker.print_progress_summary()
        return
    
    # 初始化Ray
    if args.ray_address:
        print(f"🌐 连接到Ray集群: {args.ray_address}")
        ray.init(address=args.ray_address)
    else:
        print(f"🖥️  启动本地Ray集群")
        ray.init()
    
    try:
        # 创建批量评测管理器
        manager = BatchEvaluationManager(
            config_file=args.config,
            tasks=args.tasks,
            progress_file=args.progress_file
        )
        
        # 运行批量评测
        results = manager.run_batch_evaluation(
            force_rerun=args.force_rerun,
            retry_failed_only=args.retry_failed_only
        )
        
        print(f"\n🎉 所有任务处理完成!")
        print(f"📊 进度文件: {args.progress_file}")
        print(f"📄 汇总文件: ./batch_evaluation_summaries/")
        print(f"💡 提示: 使用 --show-progress 可以随时查看进度")
        print(f"💡 提示: 使用 --retry-failed-only 可以只重跑失败的任务")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断评测")
        print(f"📊 当前进度已保存到: {args.progress_file}")
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