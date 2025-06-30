# CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval \
#   --model vllm \
#   --model_args pretrained=meta-llama/Llama-2-7b-chat-hf \
#   --tasks gsm8k \
#   --batch_size 1 \
#   --output_path results_llama2_gsm8k.json



# CUDA_VISIBLE_DEVICES=6,7 python ray_batch_eval.py \
#   --config ray_batch_eval.json \
#   --save-to-lora-path \
#   --tasks humaneval





# CUDA_VISIBLE_DEVICES=1 python ray-run_evaluation.py \
#     --lora-path /data3/user/jin509/safetunebad/ckpts/lisa/gsm8k_low_harm/checkpoint-6250 \
#     --base-model meta-llama/Llama-2-7b-chat-hf \
#     --tasks gsm8k \
#     --output /data3/user/jin509/safetunebad/ckpts/lisa/gsm8k_low_harm/checkpoint-6250/lm_eval_gsm8k.json \
#     --tensor-parallel-size 1


CUDA_VISIBLE_DEVICES=0,1,2,3 python ray-run_evaluation.py \
    --lora-path /data3/user/jin509/malicious-finetuning/experiments-sft_stage/default/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_alpaca-eval_alpaca/ \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --tasks "gsm8k","truthfulqa_mc1","truthfulqa_mc2","humaneval" \
    --output-path /data3/user/jin509/malicious-finetuning/experiments-sft_stage/default/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_alpaca-eval_alpaca/ \
    --tensor-parallel-size 4






CUDA_VISIBLE_DEVICES=1 python ray-run_evaluation.py \
    --lora-path /data3/user/jin509/safetunebad/ckpts/lisa/gsm8k_high_harm/checkpoint-6250 \
    --base-model meta-llama/Llama-2-7b-chat-hf \
    --tasks gsm8k \
    --output /data3/user/jin509/safetunebad/ckpts/lisa/gsm8k_high_harm/checkpoint-6250/lm_eval_gsm8k.json \
    --tensor-parallel-size 1



# CUDA_VISIBLE_DEVICES=1 python ray-run_evaluation.py \
#     --lora-path /data3/user/jin509/malicious-finetuning/experiments-lisa-debug-alpaca/lisa/gsm8k-BeaverTails-p0.0/Llama-2-7b-chat-hf-lora-r64-e10-b8-data5000/lisa-finetune900-rho1-benign1-align100/ \
#     --base-model meta-llama/Llama-2-7b-chat-hf \
#     --tasks gsm8k \
#     --output /data3/user/jin509/malicious-finetuning/experiments-lisa-debug-alpaca/lisa/gsm8k-BeaverTails-p0.0/Llama-2-7b-chat-hf-lora-r64-e10-b8-data5000/lisa-finetune900-rho1-benign1-align100/lm_eval_gsm8k.json \
#     --tensor-parallel-size 1



# CUDA_VISIBLE_DEVICES=1 python ray-run_evaluation.py \
#     --lora-path /data3/user/jin509/malicious-finetuning/experiments-lisa-debug/lisa/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data5000/lisa-finetune900-rho1-benign1-align100/ \
#     --base-model meta-llama/Llama-2-7b-chat-hf \
#     --tasks gsm8k \
#     --output /data3/user/jin509/malicious-finetuning/experiments-lisa-debug/lisa/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data5000/lisa-finetune900-rho1-benign1-align100/lm_eval_gsm8k.json \
#     --tensor-parallel-size 1













# HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=1,7 lm_eval \
#   --model vllm \
#   --model_args "pretrained=meta-llama/Llama-2-7b-chat-hf" \
#   --tasks humaneval \
#   --batch_size 'auto' \
#   --output_path results \
#   --confirm_run_unsafe_code \
#   --log_samples


# HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=1,7 lm_eval \
#   --model vllm \
#   --model_args "pretrained=meta-llama/Llama-3.1-8B" \
#   --tasks gsm8k \
#   --num_fewshot 0 \
#   --output_path results \
#   --confirm_run_unsafe_code \
#   --log_samples



# CUDA_VISIBLE_DEVICES=0,1 lm_eval --model hf \
#     --model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,device_map=auto" \
#     --tasks gsm8k \
#     --num_fewshot 0 \
#     --batch_size 8



# CUDA_VISIBLE_DEVICES=1,7 lm_eval --model hf \
#     --model_args "pretrained=meta-llama/Llama-3.1-8B,device_map=balanced" \
#     --tasks gsm8k \
#     --num_fewshot 0 \
#     --batch_size 8


# CUDA_VISIBLE_DEVICES=4,5,6,7 python run_evaluation.py

# CUDA_VISIBLE_DEVICES=4,5,6,7 python meval.py --task humaneval

# CUDA_VISIBLE_DEVICES=4,5,6,7 python meval.py --task mmlu

# CUDA_VISIBLE_DEVICES=4,5,6,7 python meval.py --task reasoning

# CUDA_VISIBLE_DEVICES=4,5,6,7 python meval.py --task truthfulqa



# CUDA_VISIBLE_DEVICES=4,5,6,7 python ray-run_evaluation.py \
#     --lora-path /data3/user/jin509/malicious-finetuning/experiments-debug/ptst/gsm8k-BeaverTails-p0.0/Llama-2-7b-chat-hf-lora-r64-e10-b4-data5000 \
#     --base-model meta-llama/Llama-2-7b-chat-hf \
#     --tasks humaneval \
#     --output ./ray_results \
#     --tensor-parallel-size 4 

