"""
Train bpe tokenizer
Split train/test data
encode train/test data
"""


import os
import json
from basic import Tokenizer



path = os.path.abspath("data/data.jsonl")

with open(path, 'r', encoding='utf-8') as file:
    documents = file.readlines()

data = [json.loads(d)['text'] for d in documents]

TEXT = ''.join(data)
N = int(0.8*len(TEXT))

TRAIN = TEXT[:N]
TEST = TEXT[N:]

print(len(TEST))
print(TEST[:30])

tokenizer = Tokenizer()

#tokenizer.train_encoder(TRAIN, iterations=200)
#tokenizer.save()

tokenizer.load()

train = tokenizer.encode(TRAIN)
test = tokenizer.encode(TEST)

print(len(test))

train_path = os.path.abspath("data/train.txt")
test_path = os.path.abspath("data/test.txt")

with open(train_path, 'w', encoding="utf-8") as file:
    for idx in train:
        file.write(f"{idx}\n")

with open(test_path, 'w', encoding="utf-8") as file:
    for idx in test:
        file.write(f"{idx}\n")
