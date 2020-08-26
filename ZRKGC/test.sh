#E2E
set -e


starttime=`date +'%Y-%m-%d %H:%M:%S'`

MODEL_RECOVER_PATH=model/ZRKGC_model

export CUDA_VISIBLE_DEVICES=0,1

#Five Dataset Type: wizard_random wizard_topic topical_freq topical_rare cmu_dog

for dataset in wizard_random wizard_topic topical_freq topical_rare cmu_dog;do

  DATA_DIR=test_data/${dataset}
  OUTPUT_DIR=test_log

  export PYTORCH_PRETRAINED_BERT_CACHE=unilm_v2_bert_pretrain

  #know selection
  python3 -u run_seq2seq.py  --amp --fp16  --do_predict  --num_workers 0  --tokenized_input \
  --bert_model unilm_v2_bert_pretrain --local_rank -1 \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/bert_save --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} --max_position_embeddings 256  \
  --max_seq_length 256 \
  --max_pred 40  --max_len_b 40 \
  --mask_prob 0.3 --trunc_seg a --always_truncate_tail \
  --train_batch_size 100 --eval_batch_size 500 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion_step 500 --label_smoothing 0.1 \
  --num_train_epochs 20 \
  --predict_input_file  test_${dataset}.ks.tk \
  --predict_output_file test_${dataset}.ks_score.tk

  #get ks accu and decode src file
  python3 ks_process.py ${dataset}

  #ppl
  python3 -u PPL.py --do_train --num_workers 0  --fp16 --amp --tokenized_input \
    --bert_model unilm_v2_bert_pretrain --local_rank -1 \
    --data_dir ${DATA_DIR} \
    --dev_src_file rank_test_${dataset}.src.tk --dev_tgt_file test_${dataset}.tgt.tk \
    \
    --output_dir ${OUTPUT_DIR}/bert_save --log_dir ${OUTPUT_DIR}/bert_log \
    --model_recover_path ${MODEL_RECOVER_PATH} --max_position_embeddings 256  \
    --max_seq_length 256 \
    --max_pred 40  --max_len_b 40 \
    --mask_prob 0.000001 --trunc_seg a --always_truncate_tail \
    --train_batch_size 500 --eval_batch_size 200 --gradient_accumulation_steps 1 \
    --learning_rate 0.00003 --warmup_proportion_step 500 --label_smoothing 0.1 \
    --num_train_epochs 1

  EVAL_SPLIT=${dataset}

  #decode
  python3 -u decode.py  --amp --fp16  --bert_model unilm_v2_bert_pretrain  \
    --mode s2s --need_score_traces --tokenized_input  \
    --input_file test_data/${dataset}/rank_test_${dataset}.src.tk --split ${EVAL_SPLIT}  \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --max_seq_length 256 --max_tgt_length 40 \
    --batch_size 64 --beam_size 5 --length_penalty 0 \
    --forbid_duplicate_ngrams

  echo "${MODEL_RECOVER_PATH}"."${EVAL_SPLIT}"
  python3 -u compute_bleu.py "${MODEL_RECOVER_PATH}"."${EVAL_SPLIT}" ${dataset}

  python compute_embedding_metrics.py  "${MODEL_RECOVER_PATH}"."${EVAL_SPLIT}" "test_data/${dataset}/test_${dataset}.tgt"

done

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "run time： "$((end_seconds-start_seconds))"s"