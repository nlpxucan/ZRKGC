#!/bin/bash
set -e

starttime=`date +'%Y-%m-%d %H:%M:%S'`

MODEL_RECOVER_PATH=../model/ZRKGC_model

export CUDA_VISIBLE_DEVICES=0,1

#Five Dataset Type: wizard_random wizard_topic topical_freq topical_rare cmu_dog

for dataset in wizard_random wizard_topic topical_freq topical_rare cmu_dog;do

  DATA_DIR=../test_data/${dataset}
  OUTPUT_DIR=test_log

  export PYTORCH_PRETRAINED_BERT_CACHE=../unilm_v2_bert_pretrain

  #ppl
  python3 -u PPL.py --do_train --num_workers 0  --fp16 --amp --tokenized_input \
    --bert_model ../unilm_v2_bert_pretrain --local_rank -1 \
    --data_dir ${DATA_DIR} --dev_src_file rank_test_${dataset}.src.tk --dev_tgt_file test_${dataset}.tgt.tk \
    \
    --output_dir ${OUTPUT_DIR}/bert_save --log_dir ${OUTPUT_DIR}/bert_log \
    --model_recover_path ${MODEL_RECOVER_PATH} --max_position_embeddings 256  \
    --max_seq_length 256 \
    --max_pred 40  --max_len_b 40 \
    --mask_prob 0.000001 --trunc_seg a --always_truncate_tail \
    --train_batch_size 500 --eval_batch_size 200 --gradient_accumulation_steps 1 \
    --learning_rate 0.00003 --warmup_proportion_step 500 --label_smoothing 0.1 \
    --num_train_epochs 1

done

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "run time： "$((end_seconds-start_seconds))"s"