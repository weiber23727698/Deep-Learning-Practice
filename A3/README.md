# ADL21-HW3

## training
```
python run_summarization.py \
--do_train \
--do_eval \
--model_name_or_path google/mt5-small \
--source_prefix "summarize: " \
--train_file <train_file>\
--validation_file <validation_file> \
--output_dir <output_dir> \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=1 \
--eval_accumulation_steps=4 \
--predict_with_generate \
--text_column maintext \
--summary_column title \
--num_train_epochs 2 \
--fp16 \
--max_source_length 256 \
--max_target_length 64 \
--pad_to_max_length \
--optim adafactor \
--overwrite_cache \
--overwrite_output_dir \
--cache_dir <cache_dir> \
--learning_rate 1e-3
```
* **train_file**: path to training data file. EX: ./data/train.jsonl
* **validation_file**: path to validation data file. EX: ./data/public.jsonl
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./ckpt
* **cache_dir**: The directory that store the pretrained model download from internet. Ex: ./cache

## testing
```
python run_summarization.py \
--do_predict \
--model_name_or_path <model_name_or_path> \
--source_prefix "summarize: " \
--test_file <test_file> \
--output_file <output_file> \
--output_dir <output_dir> \
--cache_dir <cache_dir> \
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
[--num_beams <num_beams>] \
[--do_sample] \
[--top_k <top_k>] \
[--top_p <top_p>] \
[--temperature <temperature>]   
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: ./ckpt
* **test_file**: path to testing data file. EX: ./data/public.jsonl
* **output_file**: Path to prediction file. EX: ./pred.jsonl
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./ckpt
* **cache_dir**: The directory that store the pretrained model download from internet. Ex: ./cache
* **num_beams**: Number of beams to use for decoding. EX: 5
* **do_sample**: Whether or not to use sampling ; use greedy decoding otherwise.
* **top_k**: The number of highest probability vocabulary tokens to keep for top-k-filtering. EX: 50
* **top_p**: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. EX: 0.9
* **temperature**: The value used to module the next token probabilities. EX: 0.75

## evaluation
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```

## Reproduce my result
```
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```