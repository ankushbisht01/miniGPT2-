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
