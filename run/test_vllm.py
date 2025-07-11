import os
import argparse

from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="VLLM推理测试")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1,
        help="张量并行大小 (默认: 1)"
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
# 使用vLLM加载模型
llm = LLM(
    model=args.model_path,
    tensor_parallel_size=args.tensor_parallel_size,
    dtype="float16",         # 使用float16以节省显存
    trust_remote_code=True
)

# 设置生成参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    top_p=0.9,
    stop=["<|eot_id|>", "</s>"]
)

print("开始推理...")

# 批量推理
outputs = llm.generate(test_examples, sampling_params)

# 打印结果
for i, output in enumerate(outputs):
    print(f"问题 {i+1}: {test_examples[i]}")
    print(f"回答: {output.outputs[0].text}")
    print("-" * 50)

print("推理完成！VLLM运行正常。")
print(f"使用模型: {args.model_path}")
print(f"张量并行大小: {args.tensor_parallel_size}")