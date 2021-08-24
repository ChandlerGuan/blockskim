OUTPUT_DIR=model/tmp/eval/debug
DATA_DIR=datasets/squad

EVAL_CKPT_DIR=model/block_skim/skim_0.001_balance_100/


# if [ -d "$OUTPUT_DIR" ]; then
#   OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
# fi

mkdir -p ${OUTPUT_DIR}

# python src/run_qa_no_trainer.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --pad_to_max_length \
#   --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log

# python src/run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --overwrite_output_dir \
#   --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log

CUDA_LAUNCH_BLOCKING=1 python src/run_squad.py \
  --model_type bert \
  --block_skim \
  --actual_skim \
  --model_name_or_path ${EVAL_CKPT_DIR} \
  --cache_name bert-base-uncased \
  --do_lower_case \
  --do_eval \
  --predict_file dev-v1.1.json \
  --data_dir ${DATA_DIR} \
  --per_gpu_eval_batch_size=16 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log