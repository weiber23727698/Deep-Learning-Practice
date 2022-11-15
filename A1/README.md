# ADL 2022 Fall Hw1

## Environment
```shell
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```
---
Above commands are no longer necessary for reproducing my result because I have done it before. 
## Intent detection
### train
```shell
python train_intent.py
```
I just use the default parameters in sameple code.

### predict
```shell
python3 test_intent.py --test_file <test_file> --ckpt_path <ckpt_path> --pred_file <pred_file>
```
* **test_file:** Path to the test file
* **ckpt_path:** Path to model checkpoint
* **pred_file:** Perdict file path

### reproduce my result
```shell
bash download.sh
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging
### train
```shell
pip install seqeval
python train_slot.py
```
I just use the default parameters in sameple code.

### predict
```shell
python3 test_slot.py --test_file <test_file> --ckpt_path <ckpt_path> --pred_file <pred_file>
```
* **test_file:** Path to the test file
* **ckpt_path:** Path to model checkpoint
* **pred_file:** Perdict file path

### reproduce my result
```shell
bash download.sh
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```
