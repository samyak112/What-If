from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset
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


def prepare_corpus(target_token_length, dataset_name, dataset_config, tokenizer, dtype=np.uint16):
    """Stream a single dataset to a .bin file of token ids. Batches tokenizer calls
    via encode_batch for throughput."""
 
    assert tokenizer.get_vocab_size() <= np.iinfo(dtype).max, f"vocab too large for {dtype}"
 
    ROOT = Path(__file__).resolve().parent
 
    file_name = f"{target_token_length}_tokens_{dataset_name.replace('/', '_')}_{dataset_config}.bin"
    bin_path = ROOT / "datasets" / file_name
 
    if bin_path.exists():
        print(f"Bin file exists for {dataset_name}/{dataset_config}, skipping")
        return bin_path
 
    print(f"Streaming {dataset_name}/{dataset_config}, target {target_token_length} tokens")
 
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )
 
    eos_id = tokenizer.token_to_id("<eos>")
 
    total_tokens = 0
    total_sequences = 0
 
    with open(bin_path, "wb") as out:
        pbar = tqdm(total=target_token_length, unit="tok")
 
        batch = []
        batch_size = 512
 
        for sample in dataset:
            batch.append(sample["text"])
 
            if len(batch) == batch_size:
                encodings = tokenizer.encode_batch(batch)
 
                for enc in encodings:
                    ids = enc.ids
                    ids.append(eos_id)
 
                    np.asarray(ids, dtype=dtype).tofile(out)
 
                    total_tokens += len(ids)
                    total_sequences += 1
                    pbar.update(len(ids))
 
                    if total_tokens >= target_token_length:
                        break
 
                batch.clear()
 
            if total_tokens >= target_token_length:
                break
 
        pbar.close()
 
    avg_seq_length = total_tokens / total_sequences if total_sequences else 0
 
    print(f"Total sequences : {total_sequences:,}")
    print(f"Total tokens    : {total_tokens:,}")
    print(f"Average length  : {avg_seq_length:.2f} tokens")
 
    if total_tokens < target_token_length:
        print(f"Warning: only got {total_tokens} tokens, wanted {target_token_length}")
 
    print(f"Wrote {total_tokens} tokens to {bin_path}")
 
    return bin_path


def prepare_multi_corpus(datasets, target_total_tokens, tokenizer, dtype=np.uint16):
    """datasets: list of (dataset_name, dataset_config, weight) tuples, weights summing to ~1.
    Streams each dataset to its own .bin file, sized by its weight share of the total budget.
    Returns dict keyed by a readable dataset key -> bin_path.
    """

    total_weight = sum(weight for _, _, weight in datasets)

    bin_paths = {}

    for dataset_name, dataset_config, weight in datasets:
        target_tokens = int(target_total_tokens * (weight / total_weight))

        bin_path = prepare_corpus(
            target_token_length=target_tokens,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dtype=dtype,
        )

        key = f"{dataset_name}/{dataset_config}"
        bin_paths[key] = bin_path

    return bin_paths


def load_tokens(bin_path, total_tokens, dtype):
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


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.tokens = torch.tensor(np.asarray(token_ids), dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length

        x = self.tokens[start:start + self.context_length]
        y = self.tokens[start + 1:start + self.context_length + 1]

        return x, y


def get_dataloaders_multi(train_datasets, val_datasets, context_length, batch_size):
    """train_datasets: dict[key -> token array], val_datasets: dict[key -> token array].
    Returns (single mixed train_loader, dict[key -> val_loader]).
    """

    train_concat = ConcatDataset([
        LanguageModelDataset(tokens, context_length=context_length)
        for tokens in train_datasets.values()
    ])

    train_loader = DataLoader(
        train_concat,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loaders = {}
    for key, tokens in val_datasets.items():
        val_dataset = LanguageModelDataset(tokens, context_length=context_length)
        val_loaders[key] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loaders


def build_dataset(config, dtype=np.uint16, tokens_per_param=20):
    """Streams/tokenizes every dataset in config.datasets, mixes them into a single
    weighted train loader, and keeps a separate val loader per dataset so val loss
    can be tracked per-source.
    """

    tokenizer = train_tokenizer(config.datasets,config.vocab_size)

    # chinchilla_scaling
    target_total_tokens = int(count_params(vocab_size=tokenizer.get_vocab_size()) * tokens_per_param)

    bin_paths = prepare_multi_corpus(
        datasets=config.datasets,
        target_total_tokens=target_total_tokens,
        tokenizer=tokenizer,
        dtype=dtype,
    )

    total_weight = sum(weight for _, _, weight in config.datasets)

    train_datasets = {}
    val_datasets = {}

    for dataset_name, dataset_config, weight in config.datasets:
        key = f"{dataset_name}/{dataset_config}"
        target_tokens = int(target_total_tokens * (weight / total_weight))

        train_tokens, val_tokens = load_tokens(bin_paths[key], total_tokens=target_tokens, dtype=dtype)

        train_datasets[key] = train_tokens
        val_datasets[key] = val_tokens

    train_loader, val_loaders = get_dataloaders_multi(
        train_datasets, val_datasets,
        context_length=config.context_length,
        batch_size=config.batch_size,
    )

    return train_loader, val_loaders, target_total_tokens, tokenizer


if __name__ == '__main__':
    config = baseConfig()

    train_loader, val_loaders, target_total_tokens, tokenizer = build_dataset(config)

    print(f"Train batches: {len(train_loader)}")
    for key, loader in val_loaders.items():
        print(f"Val ({key}) batches: {len(loader)}")