echo "Running pre-alignment LM evaluation..."
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_final.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_final_progress.csv

echo "Running pre-alignment evaluation for lm_eval..."
CUDA_VISIBLE_DEVICES=0,1,2 python run/ray_batch_eval.py --config /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_pre_alignment.json --tasks all --num-gpus 1 --progress-file /data3/user/jin509/malicious-finetuning/experiments-booster-vaccine-repnoise/lm_eval_model_configs_pre_alignment_progress.csv

