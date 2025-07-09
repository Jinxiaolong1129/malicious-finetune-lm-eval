# NOTE pre align stage
# 先生成对应的model 文件
python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise

# final model 评测
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv

# pre align model 评测
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_pre_alignment.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_pre_alignment_progress.csv



# NOTE SFT stage
python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-sft_stage
# final model 评测
# 评估"mmlu", "humaneval", "gsm8k", "truthfulqa_mc1", "truthfulqa_mc2"


# TODO
# 直接传入base_model路径这样生成的config basemodel就是路径。记得确定一下
# modelconfig保存在/data3/user/jin509/malicious-finetuning/experiments-sft_stage
python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-sft_stage --base_model "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"



# TODO
# 在run/download_data.py中设置HF_HOME
python run/download_data.py

# 设置HF_HOME
# 可以测试同时16个, 或者--num-gpus 2, 同时8个
export HF_HOME=/data3/user/jin509/.cache/huggingface
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py \
    --config /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final.json \
    --tasks all \
    --num-gpus 1 \
    --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv



CUDA_VISIBLE_DEVICES=2,3,4,5 python run/ray_batch_eval.py \
    --config /data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/lm_eval_model_configs_final-debug.json \
    --tasks humaneval \
    --num-gpus 4 \
    --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/lm_eval_model_configs_final_progress-debug.csv




CUDA_VISIBLE_DEVICES=1,2 python run/lm_eval_test.py


export HF_HOME='/data3/user/jin509/new_hf_cache'
CUDA_VISIBLE_DEVICES=1,2 python run/ray-run_evaluation.py \
  --base-model "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" \
  --lora-path "/data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/" \
  --tasks "humaneval" \
  --output-path "/home/jin509/llm_eval/lm-evaluation-harness/run" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

{
"experiment_name": "bea_epoch1",
"base_model": "meta-llama/Llama-2-7b-chat-hf",
"lora_path": "/data3/user/jin509/malicious-finetuning/experiments-sft_stage/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/checkpoint-69",
"checkpoint": 69,
"epoch": 1
},
