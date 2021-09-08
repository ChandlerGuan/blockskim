OUTPUT_DIR=model/tmp/eval/debug
TASK_NAME=SearchQA
DATA_DIR=/home/yguan/blockskim/datasets/mrqa/${TASK_NAME}

EVAL_CKPT_DIR=model/SearchQA/block_skim/bert_base/skim_0.1_balance_20_09-06-22-42/


# if [ -d "$OUTPUT_DIR" ]; then
#   OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
# fi

mkdir -p ${OUTPUT_DIR}

# python -u src/run_squad.py \
#   --model_type bert \
#   --skim_threshold 0.01 \
#   --per_gpu_eval_batch_size=1 \
#   --fast_eval 100 \
#   --model_name_or_path ${EVAL_CKPT_DIR} \
#   --cache_name bert-base-uncased \
#   --do_lower_case \
#   --do_eval \
#   --predict_file dev-v1.1.json \
#   --data_dir ${DATA_DIR} \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --overwrite_output_dir \
#   --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log

python -u src/run_mrqa.py \
  --model_type bert \
  --block_skim \
  --per_gpu_eval_batch_size 32 \
  --model_name_or_path ${EVAL_CKPT_DIR} \
  --cache_name bert-base-uncased \
  --do_lower_case \
  --do_eval \
  --predict_file dev.jsonl.gz \
  --data_dir ${DATA_DIR} \
  --max_seq_length 512 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log