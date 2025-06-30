# 先生成对应的model 文件
python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise

# final model 评测
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv

# pre align model 评测
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_pre_alignment.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_pre_alignment_progress.csv


python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-sft_stage
# final model 评测
# 评估"mmlu", "humaneval", "gsm8k", "truthfulqa_mc1", "truthfulqa_mc2"
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv
