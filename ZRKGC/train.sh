#!/bin/sh
set -e



DATA_DIR=train_data

OUTPUT_DIR=out

MODEL_RECOVER_PATH=unilm_v2_bert_pretrain/unilm1.2-base-uncased.bin


export PYTORCH_PRETRAINED_BERT_CACHE=unilm_v2_bert_pretrain
export CUDA_VISIBLE_DEVICES=2,3

  python3 -u run_seq2seq.py --do_train --num_workers 0  --fp16 --amp --tokenized_input \
  --bert_model unilm_v2_bert_pretrain --local_rank -1 --data_dir ${DATA_DIR} \
  --src_file qkr_train.src --tgt_file qkr_train.tgt --check_file qkr_train.check \
  --ks_src_file ks_train.src --ks_tgt_file ks_train.tgt \
  --output_dir ${OUTPUT_DIR}/bert_save --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} --max_position_embeddings 256  \
  --max_seq_length 256 --max_pred 40  --max_len_b 40 --train_avg_bpe_length 25 \
  --mask_prob 0.3 --trunc_seg a --always_truncate_tail --seed 42 --hidden_dropout_prob 0.11 --attention_probs_dropout_prob 0.1 \
  --train_batch_size 10 --eval_batch_size 500 --gradient_accumulation_steps 1 \
  --learning_rate 0.00003 --warmup_proportion_step 1000 --label_smoothing 0.1 \
  --num_train_epochs 20 \
  --predict_input_file qkr_dev.ks.tk --predict_output_file qkr_dev.ks_score.tk  \
