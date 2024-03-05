"""Exposing the tokenizer to a FastAPI endpoint"""


from fastapi import FastAPI
from pydantic import BaseModel

from tokenizers.bpe.basic import Tokenizer


app = FastAPI()

tokenizer = Tokenizer()
tokenizer.load()

class EncodingData(BaseModel):
    """Data for encoding/decoding"""
    text: str

class DecodingData(BaseModel):
    """Data for decoding"""
    ids : list


@app.post("/decode")
async def decode_ids(idx : DecodingData):
    """Decoding back to text"""

    ids = idx.ids
    decoded_text = tokenizer.decode(ids)
    return {"decoded_text": decoded_text}

@app.post("/encode")
async def encode_text(data : EncodingData):
    """Testing out FastAPI"""

    text = data.text
    encoded_text = tokenizer.encode(text)
    return {"encoded_text": encoded_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
