# HW2

## Enviroments
```bash
pip install -r requirements.txt
```
used only if necessary
## Context Selection
### Training
```bash
python train_multiple.py \
--model_name_or_path <model_name_or_path> \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--output_dir <output_dir> \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=2 \
--max_seq_length 512 \
--cache_dir <cache_dir> \
--pad_to_max_length \
--overwrite_output \
--train_file <train_file> \
--validation_file <validation_file> \
--context_file <context_file> \
--gradient_accumulation_steps 8
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. I use bert-base-chinese for my final result.
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/multiple-choice or ./roberta-wwm-ext/multiple_choice
* **cache_dir**: The directory that store the pretrained model download from internet
* **train_file**: path to training data file (after changing format). EX: ./temp/train.json
* **validation_file**: path to validation data file (after changing format). EX: ./temp/public.json
* **context_file**: path to the context file. EX: ./dataset/context.json


### Testing
```bash
python train_multiple.py \
--model_name_or_path <model_name_or_path> \
--do_predict \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--output_dir <output_dir> \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=2 \
--max_seq_length 512 \
--cache_dir <cache_dir> \
--pad_to_max_length \
--test_file <test_file> \
--output_file <output_file> \
--context_file <context_file> \
--gradient_accumulation_steps 8
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: ./bert/multiple-choice or ./roberta-wwm-ext/multiple_choice
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/multiple-choice or ./roberta-wwm-ext/multiple_choice
* **cache_dir**: The directory that store the pretrained model download from internet
* **test_file**: path to testing data file (after changing format) EX: ./temp/public.json or ./temp/private.json
* **context_file**: path to the context file. EX: ./temp/context.json
* **output_file**: Path to prediction file. EX: ./temp/public_context_selection_pred.json or ./temp/private_context_selection_pred.json

---
## Question Answering
### Training
```bash
python run_qa.py \
--model_name_or_path <model_name_or_path> \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir <output_dir> \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=2 \
--max_seq_length 512 \
--cache_dir <cache_dir> \
--pad_to_max_length \
--overwrite_output \
--train_file <train_file> \
--validation_file <validation_file> \
--context_file <context_file> \
--gradient_accumulation_steps 8 \
--logging_steps 500
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. I use hfl/chinese-roberta-wwm-ext for my final result. 
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/qa or ./roberta-wwm-ext/qa
* **cache_dir**: The directory that store the pretrained model download from internet
* **train_file**: path to training data file (after changing format). EX: ./temp/train.json
* **validation_file**: path to validation data file (after changing format). EX: ./temp/public.json
* **context_file**: path to the context file. EX: ./dataset/context.json

### Testing
```bash
python run_qa.py \
--model_name_or_path <model_name_or_path> \
--do_predict \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--output_dir <output_dir> \
--per_gpu_eval_batch_size=2 \
--per_device_train_batch_size=2 \
--max_seq_length 512 \
--cache_dir <cache_dir> \
--pad_to_max_length \
--test_file <test_file> \
--context_file <context_file> \
--output_file <output_file> \
--gradient_accumulation_steps 8
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: ./bert/qa or ./roberta-wwm-ext/qa
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/qa or ./roberta-wwm-ext/qa
* **cache_dir**: The directory that store the pretrained model download from internet
* **test_file**: path to testing data file (after changing format) EX: ./temp/public.json or ./temp/private.json
* **context_file**: path to the context file. EX: ./temp/context.json
* **output_file**: Path to prediction file. EX: ./public_qa_pred.json or ./private_qa_pred.json

## Reproduce my result 
```bash
bash download.sh
bash ./run.sh /path/to/context.json /path/to/public.json /path/to/pred/public.json
bash ./run.sh /path/to/context.json /path/to/private.json /path/to/pred/private.json
```
