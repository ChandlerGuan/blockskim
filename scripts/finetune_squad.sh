OUTPUT_DIR=model/tmp/

if [ -d "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_$(date +"%m-%d-%H-%M")
fi

mkdir -p ${OUTPUT_DIR}

python src/run_qa_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/log_finetune.log