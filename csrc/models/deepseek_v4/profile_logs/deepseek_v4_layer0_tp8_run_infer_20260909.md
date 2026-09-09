# DeepSeek V4 layer0 `run_infer.sh` output

## Run context

- Collected at: 2026-09-09 07:26:57 UTC
- Runtime log timestamp: 2026-09-09 15:25 (runtime logger timezone)
- Git commit: `799141ee760e762d827d8d558f7d64ca9bc73b14`
- Worktree: dirty; this run used the current uncommitted DeepSeek V4 implementation
- Command: `bash run_infer.sh`
- Exit code: `0`
- Model: `/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0`
- Device: Hygon, TP=8
- Graph: disabled
- Attention: paged KV cache with FlashAttention
- Block size: 64
- KV cache blocks: 32
- Sampling: temperature=1.0, top_p=0.8, top_k=1
- Warmup: enabled
- Max new tokens: 640
- Prompt: `山东最高的山是？`

## Runtime summary

- `_infinilm` build: succeeded, 34.184 s
- `_infinilm` install: succeeded
- Weight shards: 3/3 loaded
- Weight loading: 2997.770 ms
- Generate prefill: 9 tokens, forward 59.340 ms, total 59.861 ms
- Decode: 639 steps
- End-to-end generation: 27061.74 ms
- Natural-language check: failed; the generated text is multilingual, repetitive, and does not answer the question.

## Query

```text
<｜begin▁of▁sentence｜><｜User｜>山东最高的山是？<｜Assistant｜></think>
```

## Response

```text
 الشعاعيه يتيمه�� dével持续的ibsolutely الشعاعيه يتيمه económicservices الشعاعيه يتيمه económetricsource.com.au尷nels buruj gihabogon人居 anjara unevenlyptusTreeLabel CategoryTreeLabel有天 anjaraasadpang尷nels buruj gihabogon人居 anjarahiyon ---|---|---|---|---|---|--- kwadrado kwadrado kwadrado nalukop kasagaran etxek慌失措alerservice الشعاعيه يتيمه económicsaon behalf становништво рођено�� dével持续的几分钟kuuta半功倍}<? económicsaon behalf становништво рођено�� dével持续的izarówno sabwagaplentyaplentyaplentyaplentyTyrosineaplentyTyrosineaplentyTyrosineaplentyTyrosine burujcb licensierad豁 mediefüll dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harvestingredients kodea)SkipRAF}<? económicsaon behalf становништво рођено�� dével持续的几分钟kuuta半功 petabitskojonechernsTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika truffszer dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadikaacl出不穷 kinaug尷nels burujcb licensierad豁 dátummal华尔ahimutangKaginharian-saluran statistichesitantheraahimutangKaginharian-saluran statistichesitantrismäßerer dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadikaacl出不穷 kinaug尷nels burujcb licensierad豁 mediefüll dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika truffszer dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadikaacl出不穷 kinaug尷nels burujcb licensierad豁 mediefüll dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika trucc dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika truktur množineastern}<? económicsaon behalf становништво рођено�� dátummaledbacklogsgers basihan económicservices الشعاعيه يتيمه económicsaon behalf становништво рођено�� dével持续的几分钟kuuta半功 petabitskojonec الشعاعيه يتيمه�� dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika trucc dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika truktur množineastern}<? económicsaon behalf становништво рођено�� dátummaledbacklogsgers basihan económetricsourceAdministrazioa尷nels burujcb licensierad豁 dátummaledbacktrackingredients hilabihanKaginharian-saluran statistichesitantheraahimutangKaginhariandimensionallykeiten económicsupportive dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizonaplentyTyrosine Ngalan耳朵Noiz kontsultatua/tbsp瘫痮 kinainitsu burujabe económetricsourcehiyon}<? económicsaon behalf становништво рођено�� dátummaledbacklogsgers basihan económicsaon behalf становништво рођено�� dével持续的几分钟kuuta半功 petabitskojonec الشعاعيه يتيمه económicsaon behalf становништво рођено ентитету绎 dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizonaplentyTyrosine Ngalan耳朵Noiz kontsultatua/tbsp瘫痮 talagsaKaginharianxjzy�� proiektuakurah}<? económicsaon behalf становништво рођено ентитету绎 dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadikaacl出不穷的技巧auf尷nels burujcb licensierad豁 mediefüll dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadikaacl出不穷的技巧auf尷nels burujcb licensierad豁 mediefüll dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadikaacl出不穷的技巧auf尷nels burujcb licensierad豁 mediefüll dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika truffszer dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadikaacl出不穷 kinaug尷nels burujcb licensierad豁 mediefüll dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika trucc dévelhanced}importTreeLabel CategoryTreeLabel有天 anjara来历不明的harizon nadika truktur množineastern}<? económicsaon behalf становништво рођено�� dátummal
```

## Result

The runtime path completed without an exception, but this layer0 subset did not produce a semantically valid answer. Process success must not be treated as model-correctness success.

## Refactor regression verification

- Command: `bash run_infer.sh`
- Exit code: `0`
- `_infinilm` build: succeeded, 35.967 s
- Weight loading: 3628.982 ms
- End-to-end generation: 24368.59 ms
- Response bytes: 4495
- Response SHA-256: `ae986bf092a8a4d02da8f80d9b04d6b5ac2bc0b70cfb000e7c3cbb5d1cbeaca8`
- Comparison result: response text is byte-for-byte identical to the pre-refactor baseline.
