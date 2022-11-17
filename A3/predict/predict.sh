python run_summarization.py \
--do_predict \
--model_name_or_path ckpt5/ \
--source_prefix "summarize: " \
--test_file data/public.jsonl \
--output_file beam10_10epoch.jsonl \
--output_dir ckpt5 \
--cache_dir cache5/ \
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

python ADL22-HW3/eval.py -r /tmp2/b09902123/hw3/data/public.jsonl -s beam10_10epoch.jsonl