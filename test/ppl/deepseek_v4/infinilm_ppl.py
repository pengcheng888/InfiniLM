#!/usr/bin/env python3
"""Calculate shifted-token PPL with InfiniLM's native TP engine."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

from ppl_common import SCORING_METHOD, iter_windows, load_manifest


RESULT_SCHEMA = "deepseek_v4_infinilm_ppl_result/v1"
DEFAULT_MODEL = "/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8"
EXPECTED_MODEL_TYPE = "deepseek_v4"
EXPECTED_VOCAB_SIZE = 129280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-manifest", required=True)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--max-scored-tokens",
        type=int,
        default=127,
        help="0 scores every available target token",
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--attention", default="paged-attn")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.window < 2:
        parser.error("--window must be at least 2")
    if args.stride < 1 or args.stride >= args.window:
        parser.error("--stride must satisfy 1 <= stride < window")
    if args.max_scored_tokens < 0:
        parser.error("--max-scored-tokens must be non-negative")
    if args.tp_size < 1 or args.block_size < 1:
        parser.error("--tp-size and --block-size must be positive")
    for field in ("model", "token_manifest"):
        value = Path(getattr(args, field)).resolve()
        if not value.exists():
            parser.error(f"{field} does not exist: {value}")
        setattr(args, field, str(value))
    if args.json_output:
        args.json_output = str(Path(args.json_output).resolve())
    return args


def atomic_json(path_value: str, payload: dict) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    import infinicore
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    corpus = load_manifest(args.token_manifest)
    config = json.loads((Path(args.model) / "config.json").read_text())
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise RuntimeError(f"expected model_type={EXPECTED_MODEL_TYPE}")
    if int(config.get("vocab_size", 0)) != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"expected vocab_size={EXPECTED_VOCAB_SIZE}")
    if max(corpus.token_ids) >= EXPECTED_VOCAB_SIZE:
        raise RuntimeError("manifest contains a token outside the model vocabulary")

    target_limit = None if args.max_scored_tokens == 0 else args.max_scored_tokens
    windows = list(
        iter_windows(corpus.token_ids, args.window, args.stride, target_limit)
    )
    scored_token_count = sum(window.scored_token_count for window in windows)
    if scored_token_count < 1:
        raise RuntimeError("no tokens selected for scoring")

    num_blocks = max(2, math.ceil((args.window - 1) / args.block_size))
    precision = "W8A8" if (
        config.get("quantization_config") or config.get("compression_config")
    ) else str(config.get("torch_dtype", "unknown"))
    run_config = {
        "model": args.model,
        "precision": precision,
        "tp_size": args.tp_size,
        "attention": args.attention,
        "window": args.window,
        "stride": args.stride,
        "scored_token_count": scored_token_count,
        "manifest_sha256": corpus.manifest_sha256,
        "token_ids_sha256": corpus.token_ids_sha256,
    }
    print("INFINILM_DSV4_PPL_CONFIG " + json.dumps(run_config, sort_keys=True))

    load_start = time.perf_counter()
    model = InferEngine(
        args.model,
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(args.tp_size),
        cache_config=PagedKVCacheConfig(
            num_blocks=num_blocks, block_size=args.block_size
        ),
        enable_graph_compiling=False,
        attention_backend=args.attention,
    )
    load_model_state_dict_by_file(model, args.model, dtype=model.dtype)
    load_seconds = time.perf_counter() - load_start

    window_results = []
    nll_values = []
    infinicore.sync_device()
    scoring_start = time.perf_counter()
    for window in windows:
        input_ids = infinicore.from_list(
            [list(window.token_ids[:-1])], dtype=infinicore.int64
        )
        labels = infinicore.from_list(
            [list(window.token_ids[1:])], dtype=infinicore.int64
        )
        nll, returned_tokens = model.score_nll(
            input_ids, labels, score_start=window.prediction_start
        )
        if returned_tokens != window.scored_token_count:
            raise RuntimeError(
                f"window {window.index}: expected {window.scored_token_count} "
                f"tokens, got {returned_tokens}"
            )
        if not math.isfinite(nll) or nll < 0:
            raise RuntimeError(f"window {window.index}: invalid NLL {nll}")
        nll_values.append(nll)
        window_results.append(
            {
                "index": window.index,
                "token_start": window.token_start,
                "token_end": window.token_end,
                "scored_tokens": returned_tokens,
                "nll": nll,
            }
        )
        print(
            f"PPL window {window.index + 1}/{len(windows)} "
            f"tokens={returned_tokens} nll={nll:.6f}",
            flush=True,
        )

    infinicore.sync_device()
    scoring_seconds = time.perf_counter() - scoring_start
    total_nll = math.fsum(nll_values)
    mean_nll = total_nll / scored_token_count
    ppl = math.exp(mean_nll)
    if not math.isfinite(ppl):
        raise RuntimeError(f"non-finite PPL at mean NLL {mean_nll}")
    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        **run_config,
        "scoring_method": SCORING_METHOD,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "window_count": len(windows),
        "windows": window_results,
        "model_load_seconds": load_seconds,
        "scoring_seconds": scoring_seconds,
        "scored_tokens_per_second": scored_token_count / scoring_seconds,
    }
    if args.json_output:
        atomic_json(args.json_output, result)
    print("INFINILM_DSV4_PPL_RESULT " + json.dumps(result, sort_keys=True))
    del model
    gc.collect()
    return result


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except BaseException as error:
        print(
            "INFINILM_DSV4_PPL_ERROR "
            + json.dumps(
                {"type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
