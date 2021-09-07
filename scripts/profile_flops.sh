OUTPUT_DIR=model/tmp/eval/prof
# DATA_DIR=/home/yguan/blockskim/datasets/squad
DATA_DIR=/home/yguan/blockskim/datasets/hotpotqa

# EVAL_CKPT_DIR=model/block_skim/bert_large_wwm_new/skim_1_balance_20_seed_43/
# EVAL_CKPT_DIR=model/block_skim/skim_0.1_balance_20/
EVAL_CKPT_DIR=model/hotpotqa/albert_base/skim_0.1_balance_20/

mkdir -p ${OUTPUT_DIR}

# python -u src/run_squad_profile.py \
#   --fast_eval 100 \
python -u src/run_squad.py \
  --model_type albert \
  --block_skim \
  --actual_skim \
  --skim_threshold 0.005 \
  --per_gpu_eval_batch_size=20 \
  --model_name_or_path ${EVAL_CKPT_DIR} \
  --cache_name albert-base-v2 \
  --do_eval \
  --predict_file gold_validation.json \
  --data_dir ${DATA_DIR} \
  --max_seq_length 512 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log
