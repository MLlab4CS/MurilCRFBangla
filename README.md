## MurilCRFBangla
### Install Packages

```python
  pip3 install flair
  pip3 install seqeval
```
 
### Datasets
Please donwload the Bangla dataset from this [Link](https://multiconer.github.io/dataset) and save in the data directory

### Run the following code to train the model
```python
    !python3 train.py --dataset_path data/ \
--data_train train.txt\
--data_test test.txt\
--data_dev dev.txt\
--output_dir model \
--model_name_or_path google/muril-large-cased \
--layers -1\
--subtoken_pooling first_last\
--hidden_size 256\
--learning_rate 5e-5\
--num_epochs 30 \
--use_crf True
```
### Inference
Run the following scrpit to test the best model
```python
  python3 evaluate.py
```
