CUDA_VISIBLE_DEVICES=1,2 python run/test_vllm_lora.py \
  --base_model_path "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" \
  --lora_path "/data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/"


CUDA_VISIBLE_DEVICES=1,2 python run/test_vllm.py \
  --model_path "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" 