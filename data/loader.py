from datasets import load_dataset

def load_raw_text(split: str = "train", fraction: float = 1.0) -> str:
    # Load wikitext-103-raw-v1 from HuggingFace
    pass

def encode_corpus(text: str, tokenizer) -> list[int]:
    # Tokenize the full string in one pass
    pass