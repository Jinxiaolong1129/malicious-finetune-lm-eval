CUDA_VISIBLE_DEVICES=4,5 python ray_batch_eval.py \
  --config model_configs_debug.json \
  --save-to-lora-path \
  --tasks humaneval


# CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
#          --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
#          --tasks gsm8k \
#         --batch_size auto 