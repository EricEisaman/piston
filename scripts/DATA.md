# Browser Train data

Natural-language datasets are prepared **locally** into
`examples/piston-train-toy/static/{tokenizer,tokenized}/`.

## Prepare

```bash
# Recommended first run (manageable):
DATASETS=tinyshakespeare VOCABS=char,1024 python3 scripts/prepare-natural-data.py

# Full default set (Shakespeare + TinyStories + FineWeb-edu sample):
bash scripts/prepare-natural-data.sh
```

## Sources

| Dataset | Source |
|---|---|
| tinyshakespeare | Karpathy char-rnn `input.txt` on GitHub |
| tinystories | Hugging Face `roneneldan/TinyStories` (streaming sample) |
| fineweb | Hugging Face `HuggingFaceFW/fineweb-edu` (streaming sample) |
| tinychat | Local only: `TINYCHAT_TEXT=/path/to.txt` |

[FineWeb-2 pipeline](https://github.com/EricEisaman/fineweb-2) can produce text you then shard yourself into `static/tokenized/fineweb/`.

## Env knobs

- `DATASETS` — comma list
- `VOCABS` — `char` and/or BPE sizes (`512`…`65536`)
- `TINYSTORIES_CHARS` / `FINEWEB_CHARS` — sample size caps
- `TOKENS_PER_SHARD` — shard size (default 400000)
- `TINYCHAT_TEXT` — path for tinychat corpus
