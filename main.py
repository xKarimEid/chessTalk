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

text = data[0][:50]
print("text to encode: ")
print(text)
print("--------")

encodings = tokenizer.encode(text)
print("len of encoded ids", len(encodings))

decoded_text = tokenizer.decode(encodings)
print("decoded text")
print(decoded_text)
