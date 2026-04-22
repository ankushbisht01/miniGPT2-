import tiktoken
# from configs import vocab_size

def get_tokenizer() -> tiktoken.Encoding:
    # Load GPT-2 BPE encoding via tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    assert tokenizer.n_vocab == 50257

    return tokenizer


def validate_tokenizer(tokenizer: tiktoken.Encoding) -> None:
    text = " "
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert text == decoded
    

