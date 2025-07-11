# 设置 HuggingFace 缓存目录
export HF_HOME=/data3/user/jin509/new_hf_cache
export HF_DATASETS_CACHE=/data3/user/jin509/new_hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/data3/user/jin509/new_hf_cache

# 强制离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ALLOW_CODE_EVAL=1


# 指定 GPU 设备编号
# lm_eval --model hf \
#         --model_args pretrained=/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9,device=cuda:1 \
#         --tasks mmlu \
#         --confirm_run_unsafe_code \
#         --use_cache /data3/user/jin509/new_hf_cache


CUDA_VISIBLE_DEVICES=4,5,6,7 python run/ray-run_evaluation.py \
  --base-model "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" \
  --lora-path "/data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/" \
  --tasks "all" \
  --output-path "/home/jin509/llm_eval/lm-evaluation-harness/run" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8