CUDA_VISIBLE_DEVICES=4,5 python run/ray_batch_eval.py \
  --config model_configs_debug.json \
  --save-to-lora-path \
  --tasks humaneval


export HF_DATASETS_TRUST_REMOTE_CODE=1
CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
         --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True \
         --tasks pubmedqa \
        --batch_size auto 


CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
         --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True \
         --tasks gsm8k \
        --batch_size auto \
        --num_fewshot 0 

# pubmedqa


python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise

CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv


CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_pre_alignment.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_pre_alignment_progress.csv




python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-sft_stage

CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config model_configs_final.json --tasks gsm8k --num-gpus 1

# 评估"mmlu", "humaneval", "gsm8k", "truthfulqa_mc1", "truthfulqa_mc2"
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv


# debug 从all中删除了mmlu