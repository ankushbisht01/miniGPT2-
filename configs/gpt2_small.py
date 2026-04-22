from dataclasses import dataclass

@dataclass
class GPT2Config:
    # vocab_size: int = 50257
    # context_length: int = 256
    # n_layers: int = 12
    # n_heads: int = 12
    # d_model: int = 768
    # d_ff: int = 3072  # 4 * d_model
    # dropout: float = 0.1
    # batch_size: int = 16
    # learning_rate: float = 3e-4
    # max_epochs: int = 5
    # grad_clip: float = 1.0
    # device: str = "mps"
    # seed: int = 42
    # cache_dir: str = "data/cache"
    vocab_size: int = 50257
    context_length : int = 256
    n_layer: int = 6 #12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072            #this is the hidden layer dimension in each transformer block
    dropout: float = 0.1
    batch_size: int = 16 
    learning_rate: float = 3e-4
    max_epochs: int = 5
    grad_clip: float = 1.0
    device: str = "mps"
    seed: int = 42
    cache_dir: str = "data/cache"
    