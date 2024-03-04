"""
Tests tokenizer class
"""

from tokenizers.bpe.basic import Tokenizer
import pytest

text1 = "This is s normal text"
text2 = "Woada aøsldfkj atwefasdf æ"

@pytest.mark.parametrize("text_samples", [text1, text2])
def test_encode(text_samples):
    """Tests encoding/decoding functionality"""

    tokenizer = Tokenizer()
    tokenizer.load()


    encoded_text = tokenizer.encode(text_samples)
    decoded_text = tokenizer.decode(encoded_text)

    assert decoded_text == text_samples
