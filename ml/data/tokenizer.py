
import random
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_tokenizer(datasets,vocab_size):


    ROOT = Path(__file__).resolve().parent


    file_name = "_".join(
        name.replace("/", "_") for name, _, _ in datasets
    )

    path = ROOT / f"{file_name}.json"

    # Load existing tokenizer
    if path.exists():
        print("Loading existing tokenizer...")
        return Tokenizer.from_file(str(path))

    print("Training new tokenizer...")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<unk>",
            "<pad>",
            "<bos>",
            "<eos>"
        ]
    )

    streams = []
    weights = []

    for dataset_name, dataset_config, weight in datasets:

        dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split="train",
            streaming=True
        )

        streams.append(iter(dataset))
        weights.append(weight)

    def text_iterator():

        seen = 0
        max_samples = 1_000_000

        while seen < max_samples:

            idx = random.choices(
                range(len(streams)),
                weights=weights,
                k=1
            )[0]

            try:
                sample = next(streams[idx])
            except StopIteration:
                continue

            seen += 1

            yield sample["text"]

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer
    )

    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.save(str(path))

    print(f"Tokenizer saved to {path}")

    return tokenizer