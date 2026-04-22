import numpy as np 
from datasets import load_dataset

def load_raw_text(split: str = "train", fraction: float = 1.0) -> str:
    # Load wikitext-103-raw-v1 from HuggingFace
    dataset = load_dataset("iohadrubin/wikitext-103-raw-v1" , split=split)
    
    total_size = int(fraction * len(dataset))
    idx = np.random.randint(0,len(dataset) , size=total_size)
    result = "".join(dataset[i]["text"] for i in idx)
    return result

def encode_corpus(text: str, tokenizer) -> list[int]:
    # Tokenize the full string in one pass
    return tokenizer.encode(text)


print(load_raw_text())