"""Tokenizing the data"""

import json
from tokenizers.bpe.basic import Tokenizer


data = []
with open("data/data.jsonl", 'r', encoding="utf-8") as file:
    for line in file:
        body = json.loads(line)['text']
        data.append(body)

tokenizer = Tokenizer()

tokenizer.train_encoder(data[0], iterations = 20)
print(tokenizer.token_mapping)
