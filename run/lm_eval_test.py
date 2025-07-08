from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="vllm",
    model_args={
        "pretrained": "meta-llama/Llama-2-7b-hf",
        "lora_local_path": "/data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/",
        "tensor_parallel_size": 2,
        "dtype": "auto",
        "gpu_memory_utilization": 0.8,
        "enable_lora": True,
        "max_lora_rank": 64,
    },
    tasks=["truthfulqa_mc1"],
    batch_size="auto",
)
print(results)


# CUDA_VISIBLE_DEVICES=1 python run/ray-run_evaluation.py \
#   --base-model "meta-llama/Llama-2-7b-chat-hf" \
#   --lora-path "/data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/" \
#   --tasks "humaneval" \
#   --output-path "/home/jin509/llm_eval/lm-evaluation-harness/run" \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.8
