"""Tokenizing the data"""

import json
from tokenizers.bpe.basic import Tokenizer


data = []
with open("data/data.jsonl", 'r', encoding="utf-8") as file:
    for line in file:
        body = json.loads(line)['text']
        data.append(body)


tokenizer = Tokenizer()
tokenizer.load()

text = "Hello this is a test, probably should put this in a test folder"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
print(decoded)
