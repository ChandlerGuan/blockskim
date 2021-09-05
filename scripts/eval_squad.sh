DATA_DIR=/home/yguan/blockskim/datasets/squad

EVAL_CKPT_DIR=model/head_pruning/squad/bert_base/k_6_skim_0.1_balance_20_09-04-15-31/


# if [ -d "$OUTPUT_DIR" ]; then
#   OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
# fi


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

for LAYER_IDX in {0..11} 
do
for HEAD_IDX in {0..11} 
do
OUTPUT_DIR=model/tmp/eval/layer_${LAYER_IDX}_head_${HEAD_IDX}
mkdir -p ${OUTPUT_DIR}


CUDA_LAUNCH_BLOCKING=1 python src/run_squad.py \
  --model_type bert \
  --model_name_or_path ${EVAL_CKPT_DIR} \
  --head_pruning_idx ${HEAD_IDX} \
  --head_pruning_layer ${LAYER_IDX} \
  --do_lower_case \
  --do_eval \
  --predict_file dev-v1.1.json \
  --data_dir ${DATA_DIR} \
  --max_seq_length 512 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log

done
done
