DATA_DIR=datasets/squad


BALANCE_FACTOR=20
SKIM_FACTOR=1

for BALANCE_FACTOR in 10 20  100
do
for SKIM_FACTOR in 1 0.1 0.01 0.001
do

OUTPUT_DIR=model/block_skim/skim_${SKIM_FACTOR}_balance_${BALANCE_FACTOR}

if [ -d "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
fi

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

python src/run_squad.py \
  --model_type bert \
  --block_skim \
  --skim_factor ${SKIM_FACTOR} \
  --balance_factor ${BALANCE_FACTOR} \
  --model_name_or_path bert-base-uncased \
  --do_lower_case \
  --do_train \
  --do_eval \
  --train_file train-v1.1.json \
  --predict_file dev-v1.1.json \
  --data_dir ${DATA_DIR} \
  --per_gpu_train_batch_size 12 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log

done
done