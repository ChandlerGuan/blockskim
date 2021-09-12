# DATA_DIR=/home/yguan/blockskim/datasets/squad
DATA_DIR=/home/yguan/blockskim/datasets/hotpotqa


BALANCE_FACTOR=20
SKIM_FACTOR=0.1

# for BALANCE_FACTOR in 1  20 100
# do
# for SKIM_FACTOR in 10 1 0.1 0.01
# do

# OUTPUT_DIR=model/block_skim/bert_large_wwm/baseline
# OUTPUT_DIR=model/block_skim/bert_large_wwm/skim_${SKIM_FACTOR}_balance_${BALANCE_FACTOR}_seed_43
# OUTPUT_DIR=model/block_skim/albert_base/skim_${SKIM_FACTOR}_balance_${BALANCE_FACTOR}
OUTPUT_DIR=model/hotpotqa/albert_base/baseline

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
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --seed 43 \
  --do_train \
  --do_eval \
  --train_file gold_train.json \
  --predict_file gold_validation.json \
  --data_dir ${DATA_DIR} \
  --per_gpu_train_batch_size 12 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log

done
done
