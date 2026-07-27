#!/usr/bin/env python3
"""Download and freeze the raw WikiText-2 test split for PPL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="output UTF-8 corpus text")
    parser.add_argument("--metadata-output", required=True)
    args = parser.parse_args()

    dataset = load_dataset(DATASET, CONFIG, split=SPLIT)
    texts = list(dataset["text"])
    corpus = "\n\n".join(texts)
    raw = corpus.encode("utf-8")

    output = Path(args.output).resolve()
    metadata_output = Path(args.metadata_output).resolve()
    atomic_write(output, corpus)

    metadata = {
        "config": CONFIG,
        "dataset": DATASET,
        "dataset_fingerprint": dataset._fingerprint,
        "join_separator": "\\n\\n",
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "output": str(output),
        "row_count": len(texts),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "split": SPLIT,
        "utf8_bytes": len(raw),
    }
    atomic_write(
        metadata_output,
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
