# NOTE 下载date
# 先修改文件中的CACHE_DIR，下载data，上传data到server
python run/ray_batch_eval_download_dataset.py



# NOTE pre align stage
# 先生成对应的model 文件
python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise

# final model 评测
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --yaml-config run/ray_batch_eval_config.yaml --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_final_progress.csv

# pre align model 评测
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --yaml-config run/ray_batch_eval_config.yaml --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_pre_alignment.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_pre_alignment_progress.csv



# NOTE SFT stage
python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-sft_stage
# final model 评测
# 评估"mmlu", "humaneval", "gsm8k", "truthfulqa_mc1", "truthfulqa_mc2"
CUDA_VISIBLE_DEVICES=4,5,6,7 python run/ray_batch_eval.py --yaml-config run/ray_batch_eval_config.yaml --config xxxx.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-sft_stage/lm_eval_model_configs_pre_alignment_progress.csv



# NOTE 教程
# TODO
# 直接传入base_model路径这样生成的config basemodel就是路径。记得确定一下
# modelconfig保存在/data3/user/jin509/malicious-finetuning/experiments-sft_stage
python run/generate_config.py --root_dir /data3/user/jin509/malicious-finetuning/experiments-sft_stage --base_model "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
# NOTE 教程结束



# NOTE debug 只运行一个model

# DEBUG 可以只跑一个debug
bash run/offline.sh


CUDA_VISIBLE_DEVICES=1,2 python run/ray-run_evaluation.py \
  --base-model "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" \
  --lora-path "/data3/user/jin509/malicious-finetuning/experiments-sft_stage-back/bea/gsm8k-BeaverTails-p0.6/Llama-2-7b-chat-hf-lora-r64-e10-b8-data500-train_llama2_sys3-eval_llama2_sys3/bea/" \
  --tasks "humaneval" \
  --output-path "/home/jin509/llm_eval/lm-evaluation-harness/run" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
# NOTE debug

