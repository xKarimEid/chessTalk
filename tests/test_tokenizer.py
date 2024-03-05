"""
Tests tokenizer class
"""

import pytest
from tokenizers.bpe.basic import Tokenizer


text1 = "This is s normal text"
text2 = "Woada aøsldfkj atwefasdf æ"
text3 = ""

@pytest.mark.parametrize("text_sample", [text1, text2])
def test_encode_decode(text_sample):
    """Tests encoding/decoding functionality"""

    tokenizer = Tokenizer()
    tokenizer.load()


    encoded_text = tokenizer.encode(text_sample)
    decoded_text = tokenizer.decode(encoded_text)

    assert decoded_text == text_sample

@pytest.mark.parametrize("text_sample", [text1, text2])
def test_train(text_sample):
    """Testing training """

    tokenizer = Tokenizer()
    tokenizer.train(text_sample, vocab_size = 258)

    ids = tokenizer.encode(text_sample)
    decoded_text = tokenizer.decode(ids)

    assert decoded_text == text_sample

@pytest.mark.parametrize("text_sample", [text1, text2])
def test_save_load(text_sample):
    """Test saving and loading functionality"""

    tokenizer = Tokenizer()
    tokenizer.train(text_sample, vocab_size= 258)
    tokenizer.save()
    tokenizer.load()

    ids = tokenizer.encode(text_sample)
    decoded_text = tokenizer.decode(ids)

    assert decoded_text == text_sample
