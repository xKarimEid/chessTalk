"""Tokenizing the data"""

import os 
from tokenizers.bpe.basic import Tokenizer


train_path = os.path.abspath('data/train.txt')
test_path = os.path.abspath('data/test.txt')

with open(train_path, 'r') as file:
    train_data = file.readlines()

with open(test_path, 'r') as file:
    test_data = file.readlines()


train_data= [int(idx) for idx in train_data]


test_data = [int(idx) for idx in test_data]

tokenizer = Tokenizer()
tokenizer.load()

text = test_data[:100]
