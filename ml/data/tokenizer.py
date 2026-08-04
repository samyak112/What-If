from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_tokenizer():

    ROOT = Path(__file__).resolve().parent
    path = ROOT / "fineweb_tokenizer.json"

    # Load existing tokenizer
    if path.exists():
        print("Loading existing tokenizer...")
        return Tokenizer.from_file(str(path))

    print("Training new tokenizer...")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=16384,
        min_frequency=2,
        special_tokens=[
            "<unk>",
            "<pad>",
            "<bos>",
            "<eos>"
        ]
    )

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        split="train",
        streaming=True
    )

    def text_iterator():
        for i, sample in enumerate(dataset):
            if i >= 1_000_000:
                break
            yield sample["text"]

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer
    )

    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.save(str(path))

    print(f"Tokenizer saved to {path}")

    return tokenizer