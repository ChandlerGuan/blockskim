OUTPUT_DIR=/home/yguan/blockskim/model/tmp/eval/debug
DATA_DIR=/home/yguan/blockskim/datasets/hotpotqa

EVAL_CKPT_DIR=/home/yguan/blockskim/model/hotpotqa/block_skim_new/skim_0.1_balance_20_seed_42_09-02-11-01


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

python -u src/run_squad.py \
  --model_type bert \
  --actual_skim \
  --block_skim \
  --skim_threshold 0.1 \
  --model_name_or_path ${EVAL_CKPT_DIR} \
  --do_lower_case \
  --do_eval \
  --block_skim \
  --predict_file gold_validation.json \
  --data_dir ${DATA_DIR} \
  --per_gpu_eval_batch_size=64 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --overwrite_cache \
  --threads 1 \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log