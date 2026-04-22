# GPT-2 Reproduction: Step 1 — Data Pipeline & Tokenization

## Architectural Context
Before touching any model code, the data pipeline must be treated as a first-class citizen. GPT-2's training characteristics — BPE tokenization, fixed context windows, autoregressive targets — impose specific constraints on how you prepare data. Every decision here cascades downstream into embedding dimensions, positional encoding range, and memory layout during training.

---

## Step 1A: Dataset Selection
**Objective:**
- Select a dataset that is:
  - Small enough to iterate on quickly (M4 Pro, 16GB RAM constraint)
  - Linguistically rich enough to surface meaningful learned patterns
  - Structurally similar to GPT-2's original training distribution (natural prose, not code or tables)

**Recommended Dataset:** `wikitext-103-raw-v1` (via HuggingFace datasets)

| Dataset         | Tokens | Pros                | Cons                                 |
|----------------|--------|---------------------|--------------------------------------|
| wikitext-2     | ~2M    | Tiny, fast          | Too small to observe generalization  |
| wikitext-103   | ~103M  | Good for iteration  | Manageable on M4                     |
| OpenWebText    | ~38B   | GPT-2 original dist | Far too large for local dev          |
| TinyShakespeare| ~1M    | Classic toy         | Narrow vocab, misleading results     |

- `wikitext-103` hits the sweet spot: large enough to learn structure, small enough for reasonable training time. Use the **raw** variant to preserve original whitespace and punctuation for BPE fidelity.
- **Design consideration:** Train on a subset (e.g., first 10–20% of tokens) during early debugging. Build your pipeline to support slicing at the dataset level, not the dataloader level.

---

## Step 1B: Data Preprocessing Pipeline
**Objective:**
- Transform raw text into a flat, contiguous integer tensor suitable for sliding-window chunking.

**Key Components to Implement:**
1. `load_raw_text(split: str) -> str`
   - Loads the train/validation/test split
   - Concatenates all articles into a single string
   - **Pitfall:** Leave WikiText's = Heading = markers and @-@ hyphens in place.

2. `encode_corpus(text: str, tokenizer) -> List[int]`
   - Tokenize the full string at once (not article-by-article)
   - Returns a flat list of integer token IDs
   - **Contract:** Input is a single string. Output is a 1D sequence of integers. No padding/truncation.

3. `build_dataset(token_ids: List[int], context_length: int) -> Tuple[Tensor, Tensor]`
   - Implements sliding window (or non-overlapping block) chunking
   - **Input:** Flat sequence of N token ids, integer context_length
   - **Output:** Two tensors of shape (num_chunks, context_length): X (inputs), Y (targets)
   - **Logic:** For block [t_0, ..., t_n], X = [t_0, ..., t_{n-1}], Y = [t_1, ..., t_n]
   - **Edge case:** Drop final incomplete chunk. Do not pad.

4. `build_dataloader(dataset, batch_size, shuffle) -> DataLoader`
   - Wraps dataset in a PyTorch DataLoader
   - For training: shuffle=True; for validation: shuffle=False
   - **Memory:** With context_length=256, batch_size=16, each batch ≈ 16KB. Scale up only after validating attention implementation.

---

## Step 1C: Tokenization
**Objective:**
- Select and configure a tokenizer compatible with GPT-2's architecture (BPE via tiktoken)

**Why BPE, not WordPiece or character-level?**
- GPT-2 uses BPE. Vocab size (50,257) is baked into embedding matrix.
- BPE operates on bytes, robust to rare chars/punctuation/whitespace, no [UNK] token.
- WordPiece (BERT) is mask-LM oriented. Character-level = long sequences (bad for attention).

**Why tiktoken over HuggingFace GPT2Tokenizer?**
- tiktoken is OpenAI's reference BPE implementation, faster, produces identical token IDs, cleaner API.

**Key Properties:**
| Property         | Value    | Why it matters                                  |
|------------------|----------|------------------------------------------------|
| Vocabulary size  | 50,257   | Sets vocab_size in embedding layer (+1 is intentional)
| Context length   | 1,024    | Sets positional embedding table size            |
| Byte-level       | Yes      | No unknown tokens possible                      |
| Special tokens   | <|endoftext|> (ID: 50256) | Used as document separator, not padding |

**What to implement:**
- `get_tokenizer() -> tiktoken.Encoding`
  - Returns the gpt2 encoding object from tiktoken
  - Validate: tokenizer.n_vocab == 50257
- `validate_tokenizer(tokenizer)`
  - Round-trip test: encode/decode/compare
  - Edge cases: empty string, unicode, numbers, punctuation
  - Known ID check for reference string

---

## Expected Outputs Before Proceeding
- `tokenizer.encode("Hello, world!")` returns a list of integer IDs
- `tokenizer.decode(ids)` recovers the original string
- Full WikiText-103 train split encodes to ~103M tokens (or chosen subset)
- `X, Y = build_dataset(tokens, context_length=256)` produces tensors of shape (N, 256) with X[i, j+1] == Y[i, j]
- DataLoader iterates without error, produces batches of shape (batch_size, 256)

---

## Implementation Notes & Common Pitfalls
- Use `encode()` with allowed_special={"<|endoftext|>"} (not encode_ordinary)
- Token IDs: torch.long (int64)
- Store chunked dataset as pre-built tensor on disk (torch.save)
- Do not shuffle at token level — only at chunk level (in DataLoader)

---

## Recommended File Structure
```
gpt2-reproduction/
│
├── data/
│   ├── __init__.py
│   ├── loader.py              # load_raw_text(), encode_corpus()
│   ├── dataset.py             # build_dataset(), GPT2Dataset class
│   └── cache/                 # torch.save() artifacts go here (gitignore this)
│
├── tokenizer/
│   ├── __init__.py
│   └── bpe.py                 # get_tokenizer(), validate_tokenizer()
│
├── model/                     # Empty for now — populated in later steps
│   ├── __init__.py
│   ├── embeddings.py          # Token + positional embeddings
│   ├── attention.py           # Causal self-attention
│   ├── mlp.py                 # Feed-forward block
│   ├── block.py               # Transformer block (attn + mlp + layernorm)
│   └── gpt2.py                # Full GPT2Model assembly
│
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training loop
│   └── scheduler.py           # LR scheduling
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py             # Perplexity, loss curves
│
├── configs/
│   └── gpt2_small.py          # Model + training hyperparameters (dataclass)
│
├── utils/
│   ├── __init__.py
│   ├── logging.py             # Simple run logger
│   └── reproducibility.py     # Seed setting, determinism helpers
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   └── test_model.py          # Populated later
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── train.py                   # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Boilerplate Files

### requirements.txt
```
torch>=2.2.0
tiktoken
datasets
numpy
tqdm
```

### configs/gpt2_small.py
```python
from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    context_length: int = 256
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.1
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_epochs: int = 5
    grad_clip: float = 1.0
    device: str = "mps"
    seed: int = 42
    cache_dir: str = "data/cache"
```

### utils/reproducibility.py
```python
import torch
import random
import numpy as np

def set_seed(seed: int) -> None:
    # Set seeds across all relevant RNG sources
    pass

def get_device(preferred: str = "mps") -> torch.device:
    # Resolution priority: mps → cuda → cpu
    pass
```

### tokenizer/bpe.py
```python
import tiktoken

def get_tokenizer() -> tiktoken.Encoding:
    # Load GPT-2 BPE encoding via tiktoken
    pass

def validate_tokenizer(tokenizer: tiktoken.Encoding) -> None:
    # Round-trip test: encode -> decode -> compare
    pass
```

### data/loader.py
```python
from datasets import load_dataset

def load_raw_text(split: str = "train", fraction: float = 1.0) -> str:
    # Load wikitext-103-raw-v1 from HuggingFace
    pass

def encode_corpus(text: str, tokenizer) -> list[int]:
    # Tokenize the full string in one pass
    pass
```

### data/dataset.py
```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPT2Dataset(Dataset):
    def __init__(self, token_ids: list[int], context_length: int):
        pass
    def __len__(self) -> int:
        pass
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

def build_dataloader(dataset: GPT2Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    pass
```

### tests/test_tokenizer.py
```python
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
```

### tests/test_dataset.py
```python
import torch
import pytest
from data.dataset import GPT2Dataset, build_dataloader

def test_xy_offset():
    dummy_tokens = list(range(1000))
    context_length = 64
    dataset = GPT2Dataset(dummy_tokens, context_length)
    x, y = dataset[0]
    assert torch.all(x[1:] == y[:-1])

def test_no_incomplete_chunks():
    dummy_tokens = list(range(1000))
    context_length = 64
    dataset = GPT2Dataset(dummy_tokens, context_length)
    for i in range(len(dataset)):
        x, y = dataset[i]
        assert x.shape == (context_length,)
        assert y.shape == (context_length,)

def test_dtype():
    dummy_tokens = list(range(500))
    dataset = GPT2Dataset(dummy_tokens, 64)
    x, y = dataset[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long

def test_dataloader_batch_shape():
    dummy_tokens = list(range(2000))
    dataset = GPT2Dataset(dummy_tokens, 64)
    loader = build_dataloader(dataset, batch_size=8, shuffle=False)
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (8, 64)
    assert y_batch.shape == (8, 64)
```

### .gitignore
```
data/cache/
__pycache__/
*.pyc
.DS_Store
*.egg-info/
.env
notebooks/.ipynb_checkpoints/
```

### train.py (entry point stub)
```python
from configs.gpt2_small import GPT2Config
from utils.reproducibility import set_seed, get_device
from tokenizer.bpe import get_tokenizer, validate_tokenizer
from data.loader import load_raw_text, encode_corpus
from data.dataset import GPT2Dataset, build_dataloader

def main():
    config = GPT2Config()
    set_seed(config.seed)
    device = get_device(config.device)

    tokenizer = get_tokenizer()
    validate_tokenizer(tokenizer)

    raw_text = load_raw_text(split="train", fraction=0.1)
    token_ids = encode_corpus(raw_text, tokenizer)

    dataset = GPT2Dataset(token_ids, config.context_length)
    loader = build_dataloader(dataset, config.batch_size, shuffle=True)

    print(f"Dataset: {len(dataset)} chunks")
    print(f"Tokens per batch: {config.batch_size * config.context_length:,}")

    # Model, training loop — added in later steps

if __name__ == "__main__":
    main()
```

### Setup Commands
```bash
# Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tokenizer tests first
pytest tests/test_tokenizer.py -v

# Then dataset tests
pytest tests/test_dataset.py -v

# Full entry point smoke test
python train.py
```

---

Now implement the stubs — starting with reproducibility.py and bpe.py, then moving to loader.py and dataset.py. Share your code when ready.
