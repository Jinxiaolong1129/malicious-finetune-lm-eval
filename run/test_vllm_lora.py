import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置使用的GPU设备

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def parse_args():
    parser = argparse.ArgumentParser(description="VLLM LoRA推理测试")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True,
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_path", 
        type=str, 
        required=True,
        help="LoRA权重路径"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1,
        help="张量并行大小 (默认: 1)"
    )
    parser.add_argument(
        "--max_lora_rank", 
        type=int, 
        default=64,
        help="最大LoRA rank (默认: 64)"
    )
    return parser.parse_args()

# 直接定义5个测试例子
test_examples = [
    "What is 2 + 3?",
    "How many days are in a week?",
    "What is the capital of France?",
    "Solve: 15 × 4 = ?",
    "What color is the sky?"
]

args = parse_args()

print("加载模型...")
# 使用vLLM加载基础模型，启用LoRA
llm = LLM(
    model=args.base_model_path,
    enable_lora=True,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=0.8,
    max_lora_rank=args.max_lora_rank,
    dtype="float16",
    trust_remote_code=True
)

# 创建LoRA请求
lora_request = LoRARequest("lora", 1, args.lora_path)

# 设置生成参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    top_p=0.9,
    stop=["<|eot_id|>", "</s>"]
)

print("开始推理...")

# 批量推理，使用LoRA
outputs = llm.generate(
    test_examples, 
    sampling_params,
    lora_request=lora_request
)

# 打印结果
for i, output in enumerate(outputs):
    print(f"问题 {i+1}: {test_examples[i]}")
    print(f"回答: {output.outputs[0].text}")
    print("-" * 50)

print("推理完成！VLLM LoRA运行正常。")
print(f"使用基础模型: {args.base_model_path}")
print(f"使用LoRA权重: {args.lora_path}")
print(f"张量并行大小: {args.tensor_parallel_size}")
print(f"最大LoRA rank: {args.max_lora_rank}")


