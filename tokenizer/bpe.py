import tiktoken

def get_tokenizer() -> tiktoken.Encoding:
    # Load GPT-2 BPE encoding via tiktoken
    pass

def validate_tokenizer(tokenizer: tiktoken.Encoding) -> None:
    # Round-trip test: encode -> decode -> compare
    pass