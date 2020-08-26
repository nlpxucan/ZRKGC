#!/bin/sh
set -e



DATA_DIR=train_data

OUTPUT_DIR=out

MODEL_RECOVER_PATH=unilm_v2_bert_pretrain/unilm1.2-base-uncased.bin


export PYTORCH_PRETRAINED_BERT_CACHE=unilm_v2_bert_pretrain
export CUDA_VISIBLE_DEVICES=0,1,2,3

  python3 -u run_seq2seq.py --do_train --num_workers 0  --fp16 --amp --tokenized_input \
  --bert_model unilm_v2_bert_pretrain --local_rank -1 \
  --data_dir ${DATA_DIR} \
  --src_file qkr_train.src --tgt_file qkr_train.tgt --check_file qkr_train.check --style_file qkr_train.style \
  --dev_src_file  qkr_dev.src --dev_tgt_file qkr_dev.tgt --dev_check_file qkr_dev.check --dev_style_file qkr_dev.style \
  --ks_src_file ks_train.src --ks_tgt_file ks_train.tgt \
  --ks_dev_src_file  ks_dev.src --ks_dev_tgt_file ks_dev.tgt \
  --output_dir ${OUTPUT_DIR}/bert_save --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} --max_position_embeddings 256  \
  --max_seq_length 256 \
  --max_pred 40  --max_len_b 40 \
  --mask_prob 0.35 --trunc_seg a --always_truncate_tail \
  --train_batch_size 10 --eval_batch_size 500 --gradient_accumulation_steps 1 \
  --learning_rate 0.00003 --warmup_proportion_step 1000 --label_smoothing 0.1 \
  --num_train_epochs 20 \
  --predict_input_file test_qkr_dev.ks.tk --predict_output_file test_qkr_dev.ks_score.tk  \
