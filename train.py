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