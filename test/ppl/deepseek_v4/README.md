# DeepSeek V4 PPL

This directory evaluates causal, shifted-token perplexity with InfiniLM's
native C++/InfiniCore TP engine. It does not use a Torch, Transformers or
SGLang model-forward fallback.

The runner computes:

```text
mean_nll = sum(-log p(x_t | x_<t)) / scored_tokens
ppl      = exp(mean_nll)
```

Overlapping windows provide left context, while every selected target token is
scored exactly once. `score_nll` disables generation graph replay only for this
quality test and computes FP32 token losses on the rank-0 InfiniCore stream.

## Environment

Run inside `wangpengcheng_sglang`:

```bash
source ~/.bashrc
source /.myenv.sh
cd /workspace_codex/InfiniLM
export PYTHONPATH=/workspace_codex/InfiniCore/python:/workspace_codex/InfiniLM/python:${PYTHONPATH}
```

## Prepare a bounded WikiText-2 manifest

Tokenize once with the model tokenizer. Reuse the same manifest for every
backend or precision being compared.

```bash
python test/ppl/deepseek_v4/prepare_ppl_corpus.py \
  --input /path/to/wikitext-2-raw-v1-test.txt \
  --tokenizer /data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8 \
  --max-tokens 128 \
  --output /workspace_codex/ppl_wikitext2_dsv4_128.json
```

## InfiniLM TP8 smoke PPL

Check `hy-smi --showpids` first and do not start while another workload owns
the GPUs.

```bash
timeout 3600 python -u test/ppl/deepseek_v4/infinilm_ppl.py \
  --model /data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8 \
  --token-manifest /workspace_codex/ppl_wikitext2_dsv4_128.json \
  --window 128 \
  --stride 64 \
  --max-scored-tokens 127 \
  --tp-size 8 \
  --attention paged-attn \
  --json-output /workspace_codex/ppl_dsv4_w8a8_128.json
```

Set `--max-scored-tokens 0` to score the full manifest. PPL scoring throughput
is diagnostic and must not be reported as normal generation performance.
