"""Download GSM8K and write standard single-turn message JSONL files."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/gsm8k"))
    args = parser.parse_args()
    dataset = load_dataset("openai/gsm8k", "main", revision=GSM8K_REVISION)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        with (args.output / f"{split}.jsonl").open("w") as stream:
            for row in dataset[split]:
                record = {"messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(dataset[split])} examples -> {args.output / (split + '.jsonl')}")
    (args.output / "source.json").write_text(json.dumps({
        "dataset": "openai/gsm8k", "config": "main", "revision": GSM8K_REVISION,
        "splits": {split: len(dataset[split]) for split in ("train", "test")},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
