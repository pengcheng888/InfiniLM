#!/usr/bin/env python3
"""Tokenize a UTF-8 corpus once and freeze it for comparable PPL runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ppl_common import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UTF-8 corpus text")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer directory")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="truncate after tokenization; 0 keeps all tokens",
    )
    args = parser.parse_args()
    if args.max_tokens < 0 or args.max_tokens == 1:
        parser.error("--max-tokens must be 0 or at least 2")

    source_path = Path(args.input).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    raw = source_path.read_bytes()
    text = raw.decode("utf-8")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True
    )
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if args.max_tokens:
        token_ids = token_ids[: args.max_tokens]
    payload = build_manifest(
        token_ids,
        source=str(source_path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        tokenizer=str(tokenizer_path),
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "token_count": payload["token_count"],
                "token_ids_sha256": payload["token_ids_sha256"],
                "manifest_sha256": payload["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
