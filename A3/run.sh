export CUDA_VISIBLE_DEVICES=0

python run_summarization.py \
--do_predict \
--model_name_or_path ./ckpt/ \
--source_prefix "summarize: " \
--test_file "$1" \
--output_file "$2" \
--output_dir ckpt/ \
--cache_dir ./cache/ \
--per_device_eval_batch_size=1 \
--eval_accumulation_steps=4 \
--predict_with_generate \
--text_column maintext \
--summary_column title \
--fp16 \
--max_source_length 256 \
--max_target_length 64 \
--pad_to_max_length \
--optim adafactor \
--num_beams 10