DATA_DIR=datasets/squad

EVAL_CKPT_DIR={YOUR_CKPT}

python -u src/run_squad.py \
  --model_type bert \
  --model_name_or_path ${EVAL_CKPT_DIR} \
  --do_lower_case \
  --do_eval \
  --block_skim \
  --actual_skim \
  --predict_file dev-v1.1.json \
  --data_dir ${DATA_DIR} \
  --max_seq_length 512 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log
