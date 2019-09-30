BERT_BASE_DIR=uncased_L-12_H-768_A-12
EXPTS_DIR=/tmp
TFRecords=/tmp/TFRecords/mentions/train
USE_TPU=true
TPU_NAME=tpu0

EXP_NAME=BERT_fntn
INIT=$BERT_BASE_DIR/bert_model.ckpt 

python run_classifier.py \
  --do_train=true \
  --do_eval=false \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --max_seq_length=256 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_cands=64 \
  --save_checkpoints_steps=6000 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME

