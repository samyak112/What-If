from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs import baseConfig
from .tokenizer import train_tokenizer


def count_params(vocab_size):
    from ..models.base.transformer import BaseTransformer

    config = baseConfig()

    model = BaseTransformer(
        d_model=config.d_model,
        n_heads=config.heads,
        dropout=config.dropout,
        num_layers=config.layers,
        vocab_size=vocab_size,
        max_seq_len=1024,
    )

    total = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        print(f"{name:70s} {count/1e6:.3f}M")

    print(f"\nTotal: {total/1e6:.3f}M")

    return total


def prepare_corpus(target_token_length, tokenizer, dataset_name="HuggingFaceFW/fineweb", dataset_config="sample-10BT",dtype=np.uint16):

    assert tokenizer.get_vocab_size() <= np.iinfo(dtype).max, f"vocab too large for {dtype}"

    ROOT = Path(__file__).resolve().parent

    file_name = f"{target_token_length}_tokens_{dataset_name.replace("/", "_")}.bin"
    bin_path = ROOT/ "datasets" / file_name

    if bin_path.exists():
        print("Bin file exists, skipping")
        return bin_path

    print(f"Streaming corpus, target {target_token_length} tokens")

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )

    eos_id = tokenizer.token_to_id("<eos>")

    total_tokens = 0

    with open(bin_path, "wb") as out:

        pbar = tqdm(total=target_token_length, unit="tok")
        for sample in dataset:
            ids = tokenizer.encode(sample["text"]).ids
            ids.append(eos_id)

            np.asarray(ids, dtype=np.uint16).tofile(out)
            total_tokens += len(ids)

            pbar.update(len(ids))

            if total_tokens >= target_token_length:
                break
        pbar.close()

    if total_tokens < target_token_length:
        print(f"Warning: only got {total_tokens} tokens, wanted {target_token_length}")

    print(f"Wrote {total_tokens} tokens to {bin_path}")

    return bin_path


def load_tokens(bin_path,total_tokens,dtype):
    token_ids = np.memmap(
        bin_path,
        dtype=dtype,
        mode="r",
    )

    token_ids = token_ids[:total_tokens]

    split = int(len(token_ids) * 0.9)

    train_tokens = token_ids[:split]
    val_tokens = token_ids[split:]

    return train_tokens, val_tokens


def get_dataloaders(train_tokens, val_tokens, context_length, batch_size):
    train_dataset = LanguageModelDataset(train_tokens, context_length=context_length)
    val_dataset = LanguageModelDataset(val_tokens, context_length=context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length

        x = self.tokens[start:start + self.context_length]
        y = self.tokens[start + 1:start + self.context_length + 1]

        return x, y

def build_dataset(config, dataset_name="HuggingFaceFW/fineweb", dataset_config="sample-10BT", dtype=np.uint16,tokens_per_param=20):

    tokenizer = train_tokenizer()

    # chinchilla_scaling
    target_tokens = int(count_params(vocab_size=tokenizer.get_vocab_size()) * tokens_per_param)

    bin_path = prepare_corpus(
        target_token_length=target_tokens,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dtype=dtype,
    )

    print(bin_path)

    train_tokens, val_tokens = load_tokens(bin_path, total_tokens=target_tokens, dtype=dtype)

    train_loader, val_loader = get_dataloaders(
        train_tokens, val_tokens,
        context_length=config.context_length,
        batch_size=config.batch_size,
    )

    return train_loader, val_loader,target_tokens, tokenizer

if __name__ == '__main__':

    ROOT = Path(__file__).resolve().parent

    tokenizer = train_tokenizer()

    config = baseConfig()

    total_tokens = count_params(tokenizer.get_vocab_size())
    prepare_corpus(target_token_length=total_tokens,tokenizer=tokenizer)
