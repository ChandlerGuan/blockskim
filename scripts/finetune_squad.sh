DATA_DIR=datasets/squad

BALANCE_FACTOR=20
SKIM_FACTOR=0.1

python src/run_squad.py \
  --model_type bert \
  --skim_factor ${SKIM_FACTOR} \
  --balance_factor ${BALANCE_FACTOR} \
  --pruning_k ${PRUNING_K} \
  --model_name_or_path bert-base-uncased \
  --seed 42 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --train_file gold_train.json \
  --predict_file gold_validation.json \
  --data_dir ${DATA_DIR} \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
