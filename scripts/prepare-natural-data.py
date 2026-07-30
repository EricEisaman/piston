#!/usr/bin/env python3
"""
Prepare Browser Train natural-language datasets locally.

Sources (local prep only — no third-party demo CDN):
  - TinyShakespeare: Karpathy char-rnn input.txt (GitHub raw)
  - TinyStories: Hugging Face roneneldan/TinyStories (optional; needs `datasets`)
  - FineWeb subset: Hugging Face HuggingFaceFW/fineweb-edu (optional; streaming sample)
  - TinyChat: skipped unless TINYCHAT_TEXT= path to a local .txt is provided

Writes llm.c-style uint16 shards (magic 20251003) and HF tokenizer.json files under:
  examples/piston-train-toy/static/{tokenizer,tokenized}/

Usage:
  python3 scripts/prepare-natural-data.py
  DATASETS=tinyshakespeare,tinystories VOCABS=char,1024 python3 scripts/prepare-natural-data.py
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

MAGIC = 20251003
VERSION = 1
HEADER_INTS = 256
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "examples" / "piston-train-toy" / "static"
TOKENIZER_OUT = STATIC / "tokenizer"
TOKENIZED_OUT = STATIC / "tokenized"
CACHE = ROOT / ".cache" / "browser-train-data"

DEFAULT_DATASETS = ["tinyshakespeare", "tinystories", "fineweb"]
DEFAULT_VOCABS = ["char", "512", "1024", "2048", "4096", "8192"]
TOKENS_PER_SHARD = int(os.environ.get("TOKENS_PER_SHARD", "400000"))
VAL_FRACTION = 0.05
FINEWEB_CHARS = int(os.environ.get("FINEWEB_CHARS", "5_000_000"))
TINYSTORIES_CHARS = int(os.environ.get("TINYSTORIES_CHARS", "8_000_000"))


def write_shard(path: Path, token_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ntok = len(token_ids)
    header = [0] * HEADER_INTS
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = ntok
    with path.open("wb") as f:
        f.write(struct.pack(f"<{HEADER_INTS}i", *header))
        f.write(struct.pack(f"<{ntok}H", *token_ids))
    print(f"  wrote {path.relative_to(ROOT)} ({ntok} tokens)")


def write_shards(dataset: str, vocab_label: str, token_ids: list[int]) -> None:
    if not token_ids:
        raise SystemExit(f"no tokens for {dataset}/{vocab_label}")
    split_at = max(1, int(len(token_ids) * (1.0 - VAL_FRACTION)))
    train_ids = token_ids[:split_at]
    val_ids = token_ids[split_at:] or token_ids[-min(1024, len(token_ids)) :]

    def dump(split: str, ids: list[int]) -> None:
        for i in range(0, len(ids), TOKENS_PER_SHARD):
            chunk = ids[i : i + TOKENS_PER_SHARD]
            idx = i // TOKENS_PER_SHARD
            out = TOKENIZED_OUT / dataset / vocab_label / f"{split}-{idx:04d}.bin"
            write_shard(out, chunk)

    dump("train", train_ids)
    dump("val", val_ids[: min(len(val_ids), TOKENS_PER_SHARD)])


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cache hit {dest}")
        return dest
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_shakespeare() -> str:
    path = download(SHAKESPEARE_URL, CACHE / "tinyshakespeare.txt")
    return path.read_text(encoding="utf-8", errors="replace")


def load_text_file(path: Path, max_chars: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None:
        return text[:max_chars]
    return text


def load_hf_text(dataset_id: str, text_field: str, max_chars: int, split: str = "train") -> str:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Hugging Face `datasets` is required for this source.\n"
            "  pip install datasets tokenizers\n"
            f"Original error: {e}"
        ) from e

    print(f"  streaming {dataset_id} (target ~{max_chars:,} chars)")
    ds = load_dataset(dataset_id, split=split, streaming=True)
    parts: list[str] = []
    total = 0
    for row in ds:
        chunk = row.get(text_field) or ""
        if not isinstance(chunk, str) or not chunk.strip():
            continue
        parts.append(chunk.strip())
        total += len(chunk)
        if total >= max_chars:
            break
    text = "\n\n".join(parts)
    return text[:max_chars]


def encode_char(text: str) -> list[int]:
    # Match Browser Train char tokenizer: bytes 0..255 + <eos> at 256
    eos = 256
    ids = [min(255, ord(ch) if ord(ch) < 256 else ord("?")) for ch in text]
    # Append eos occasionally at paragraph breaks for LM practice
    out: list[int] = []
    for i, tid in enumerate(ids):
        out.append(tid)
        if text[i] == "\n" and i + 1 < len(text) and text[i + 1] == "\n":
            out.append(eos)
    if not out or out[-1] != eos:
        out.append(eos)
    if max(out) > 65535:
        raise SystemExit("char token id overflow")
    return out


def train_and_encode_bpe(text: str, vocab_size: int, out_dir: Path) -> list[int]:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as e:
        raise SystemExit(
            "Python package `tokenizers` is required for BPE vocabs.\n"
            "  pip install tokenizers\n"
            f"Original error: {e}"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    special = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special,
        show_progress=True,
        min_frequency=2,
    )
    # Train from an iterator of lines
    lines = [ln for ln in text.splitlines() if ln.strip()] or [text]
    tokenizer.train_from_iterator(lines, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "mask_token": "<mask>",
        "model_max_length": 2048,
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  wrote {tokenizer_json.relative_to(ROOT)}")

    # Encode full corpus without re-adding bos/eos on every line: encode raw
    raw = Tokenizer.from_file(str(tokenizer_json))
    raw.post_processor = None
    enc = raw.encode(text)
    ids = enc.ids
    eos_id = raw.token_to_id("<eos>")
    if eos_id is not None and (not ids or ids[-1] != eos_id):
        ids = list(ids) + [eos_id]
    if max(ids) > 65535:
        raise SystemExit(f"token id overflow for vocab {vocab_size}")
    return list(ids)


def prepare_dataset(name: str, text: str, vocabs: list[str]) -> None:
    print(f"==> {name} ({len(text):,} chars)")
    for vocab in vocabs:
        if vocab == "char":
            print(f"  encoding char")
            ids = encode_char(text)
            write_shards(name, "char", ids)
            continue
        size = int(vocab)
        print(f"  training BPE vocab={size}")
        ids = train_and_encode_bpe(text, size, TOKENIZER_OUT / name / vocab)
        write_shards(name, vocab, ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=os.environ.get("DATASETS", ",".join(DEFAULT_DATASETS)),
        help="comma-separated: tinyshakespeare,tinystories,fineweb,tinychat",
    )
    parser.add_argument(
        "--vocabs",
        default=os.environ.get("VOCABS", ",".join(DEFAULT_VOCABS)),
        help="comma-separated: char,512,1024,...",
    )
    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    vocabs = [v.strip() for v in args.vocabs.split(",") if v.strip()]

    CACHE.mkdir(parents=True, exist_ok=True)
    TOKENIZER_OUT.mkdir(parents=True, exist_ok=True)
    TOKENIZED_OUT.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        if name == "tinyshakespeare":
            prepare_dataset(name, load_shakespeare(), vocabs)
        elif name == "tinystories":
            cache_txt = CACHE / "tinystories.txt"
            if cache_txt.exists():
                text = load_text_file(cache_txt, TINYSTORIES_CHARS)
            else:
                text = load_hf_text("roneneldan/TinyStories", "text", TINYSTORIES_CHARS)
                cache_txt.write_text(text, encoding="utf-8")
            prepare_dataset(name, text, vocabs)
        elif name == "fineweb":
            cache_txt = CACHE / "fineweb-edu-sample.txt"
            if cache_txt.exists():
                text = load_text_file(cache_txt, FINEWEB_CHARS)
            else:
                # fineweb-edu is English, manageable sample via streaming
                text = load_hf_text("HuggingFaceFW/fineweb-edu", "text", FINEWEB_CHARS)
                cache_txt.write_text(text, encoding="utf-8")
            prepare_dataset(name, text, vocabs)
        elif name == "tinychat":
            custom = os.environ.get("TINYCHAT_TEXT")
            if not custom:
                print(
                    "SKIP tinychat: set TINYCHAT_TEXT=/path/to/chat.txt "
                    "(one conversation/document per blank-line block)",
                    file=sys.stderr,
                )
                continue
            prepare_dataset(name, load_text_file(Path(custom)), vocabs)
        else:
            raise SystemExit(f"unknown dataset: {name}")

    print("Done. Assets under examples/piston-train-toy/static/{tokenizer,tokenized}")


if __name__ == "__main__":
    main()
