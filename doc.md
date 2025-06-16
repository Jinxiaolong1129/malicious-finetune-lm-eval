# Ray批量评测脚本使用说明

## 生成model_configs.json

python generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments --output model_configs.json


## 配置文件示例 (model_configs_debug.json)

```json
[
    {
        "experiment_name": "bea_epoch10",
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "lora_path": "/data3/user/jin509/malicious-finetuning/experiments-back/bea/gsm8k-BeaverTails-p0.2/Llama-2-7b-chat-hf-lora-r64-e10-b16-data100/bea/checkpoint-70",
        "checkpoint": 70,
        "epoch": 10
    },
    {
        "experiment_name": "default_epoch1",
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "lora_path": "/data3/user/jin509/malicious-finetuning/experiments-back/default/gsm8k-BeaverTails-p0.2/Llama-2-7b-chat-hf-lora-r64-e10-b16-data100/checkpoint-7",
        "checkpoint": 7,
        "epoch": 1
    },
    {
        "experiment_name": "ptst_final",
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "lora_path": "/data3/user/jin509/malicious-finetuning/experiments-back/ptst/gsm8k-BeaverTails-p0.2/Llama-2-7b-chat-hf-lora-r64-e10-b16-data100",
        "checkpoint": "final",
        "epoch": "final"
    }
]
```

## 进度文件结构 (lm_eval_experiment_progress.csv)

脚本会自动创建和维护一个CSV文件来追踪每个任务的状态：

| experiment_name | status | start_time | end_time | duration_minutes | score | base_model | lora_path | output_file | log_file | error_message | worker_pid | gpu_id | retry_count | tasks | created_time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bea_epoch10 | completed | 2025-06-16T14:30:25 | 2025-06-16T15:45:10 | 74.75 | 0.6524 | /path/to/llama2-7b-hf | /path/to/math-lora-v1 | /path/to/results.json | /path/to/eval.log | | 12345 | 3 | 0 | humaneval | 2025-06-16T14:29:45 |

## 基本使用方法

### 环境准备
```bash
export HF_HOME=/path/to/your/local/hf_cache
# 下载数据集到这里，和model
```


### 1. 首次运行
```bash
# 基本运行
python ray_batch_eval.py --config config.json

# 保存结果到各自的LoRA目录
python ray_batch_eval.py --config config.json --save-to-lora-path

# 指定评测任务
python ray_batch_eval.py --config config.json --tasks "humaneval" --save-to-lora-path

# 评测所有任务 humaneval, mmlu, truthfulqa_mc1, truthfulqa_mc2, gsm8k
python ray_batch_eval.py --config config.json --tasks "all" --save-to-lora-path

# 自定义进度文件位置
python ray_batch_eval.py --config config.json --progress-file "./experiments/eval_progress.csv"
```

### 2. 查看进度
```bash
# 只查看当前进度，不执行评测
python ray_batch_eval.py --config config.json --show-progress
```

### 3. 断点续传
```bash
# 程序被中断后，直接重新运行即可自动跳过已完成的任务
python ray_batch_eval.py --config config.json

# 只重跑失败的任务
python ray_batch_eval.py --config config.json --retry-failed-only

# 强制重跑所有任务（忽略之前的进度）
python ray_batch_eval.py --config config.json --force-rerun
```


## 进度管理特性

### 状态类型
- **pending**: 待执行
- **running**: 正在运行
- **completed**: 成功完成
- **failed**: 失败
- **timeout**: 超时
- **error**: 异常


### 智能恢复机制
1. **已完成任务**: 自动跳过，除非使用 `--force-rerun`
2. **失败任务**: 自动重试
3. **运行中任务**: 如果程序异常退出，重启后会重新执行
4. **新任务**: 配置文件中新增的任务会自动加入待执行列表



## 实际使用流程示例

### 场景1: 大规模批量评测

```bash
# 第1天：启动评测
python ray_batch_eval.py --config large_config.json --progress-file day1_progress.csv

# 程序运行了一夜，完成了50%的任务

# 第2天：查看进度
python ray_batch_eval.py --config large_config.json --progress-file day1_progress.csv --show-progress

# 继续剩余任务
python ray_batch_eval.py --config large_config.json --progress-file day1_progress.csv

# 如果发现有些任务失败，只重跑失败的
python ray_batch_eval.py --config large_config.json --progress-file day1_progress.csv --retry-failed-only
```

### 场景2: 添加新实验

```bash
# 原有的config.json已经跑过一些任务
python ray_batch_eval.py --config config.json

# 现在config.json中添加了新的实验配置
# 再次运行时，只会执行新添加的任务
python ray_batch_eval.py --config config.json
```


## 输出文件说明

### 1. 进度文件
- **位置**: `lm_eval_experiment_progress.csv`（可自定义）
- **内容**: 每个任务的详细状态信息
- **更新**: 实时更新，任务开始/结束时立即写入

### 2. 评测结果文件
- **集中模式**: `./batch_evaluation_results/experiment_name_humaneval_timestamp.json`
- **分布模式**: `{lora_path}/evaluation_humaneval_timestamp.json`

### 3. 日志文件
- **集中模式**: `./batch_evaluation_results/log_lm_eval/experiment_name_humaneval_timestamp.log`
- **分布模式**: `{lora_path}/log_lm_eval/experiment_name_humaneval_timestamp.log`

### 4. 汇总文件
- **位置**: `./batch_evaluation_summaries/batch_evaluation_summary_timestamp.json`
- **内容**: 整个批次的汇总信息和所有任务的详细结果

## 常见问题

### Q: 如何重新运行特定的失败任务？
A: 可以编辑进度CSV文件，将对应任务的status改为"pending"，或者使用`--retry-failed-only`参数。

### Q: 进度文件损坏了怎么办？
A: 删除进度文件，程序会重新创建并从头开始执行所有任务。

### Q: 如何暂停正在运行的评测？
A: 使用Ctrl+C中断程序，当前进度会自动保存。重新运行时会从中断处继续。

### Q: 如何查看实时运行状态？
A: 可以在另一个终端中使用`--show-progress`参数查看当前进度，或直接查看CSV文件。