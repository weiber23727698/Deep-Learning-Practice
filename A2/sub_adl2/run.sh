# predict multiple choice
python train_multiple.py \
--model_name_or_path submit/multiple_choice \
--do_predict \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--output_dir submit/multiple_choice \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=2 \
--max_seq_length 512 \
--cache_dir submit/cache_mul \
--pad_to_max_length \
--test_file "${2}" \
--output_file map2.json \
--context_file "${1}" \
--gradient_accumulation_steps 8

# predict QA
python run_qa.py \
--model_name_or_path submit/qustion_answering \
--do_predict \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--output_dir submit/qustion_answering \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=2 \
--max_seq_length 512 \
--cache_dir submit/cache_qa \
--pad_to_max_length \
--test_file map2.json \
--context_file "${1}" \
--output_file "${3}" \
--gradient_accumulation_steps 8
