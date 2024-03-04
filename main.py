"""Exposing the tokenizer to an endpoint"""


from fastapi import FastAPI
from pydantic import BaseModel

from tokenizers.bpe.basic import Tokenizer


app = FastAPI()

tokenizer = Tokenizer()
tokenizer.load()

class myData(BaseModel):
    text: str


@app.post("/encode")
async def encode_text(data : myData):
    """Testing out FastAPI"""

    text = data.text
    print(text)

    encoded_text = tokenizer.encode(text)
    return {"encoded_text": encoded_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
