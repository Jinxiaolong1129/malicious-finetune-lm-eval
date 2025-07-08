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
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv



CUDA_VISIBLE_DEVICES=3 python run/ray-run_evaluation.py \
  --base-model "meta-llama/Llama-2-7b-chat-hf" \
  --lora-path "/data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_alpaca-eval_alpaca/bea/" \
  --tasks "humaneval" \
  --output-path "/home/jin509/llm_eval/lm-evaluation-harness/run" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9

{
"experiment_name": "bea_epoch1",
"base_model": "meta-llama/Llama-2-7b-chat-hf",
"lora_path": "/data3/user/jin509/malicious-finetuning/experiments-sft_stage/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/checkpoint-69",
"checkpoint": 69,
"epoch": 1
},
