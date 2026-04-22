import pytest
from tokenizer.bpe import get_tokenizer, validate_tokenizer

def test_vocab_size():
    tokenizer = get_tokenizer()
    assert tokenizer.n_vocab == 50257

def test_roundtrip():
    tokenizer = get_tokenizer()
    text = "Hello, world! This is a GPT-2 reproduction."
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_known_ids():
    tokenizer = get_tokenizer()
    assert tokenizer.encode("Hello world") == [15496, 995]

def test_empty_string():
    tokenizer = get_tokenizer()
    assert tokenizer.encode("") == []