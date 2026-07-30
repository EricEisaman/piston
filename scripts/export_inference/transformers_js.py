"""Emit a Transformers.js–style package from a Browser Train model card + ONNX graph."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _decoder_lm_with_attention_mask():
    """Lazy wrapper class (requires torch)."""
    import torch
    from torch import nn

    from .models import DecoderLM

    class DecoderLMWithAttentionMask(nn.Module):
        """Wrap DecoderLM so the ONNX graph accepts input_ids + attention_mask (mask unused)."""

        def __init__(self, inner: DecoderLM):
            super().__init__()
            self.inner = inner
            self.block_size = inner.block_size

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            _ = attention_mask
            return self.inner(input_ids)

    return DecoderLMWithAttentionMask


def _block_size(card: dict[str, Any], key: str = "target") -> int:
    block = card.get("blockSize")
    if isinstance(block, dict):
        return max(int(block.get(key) or block.get("source") or block.get("target") or 64), 1)
    return max(int(block or 64), 1)


def build_char_tokenizer_json(card: dict[str, Any]) -> dict[str, Any]:
    """Build a HuggingFace tokenizers WordLevel JSON for char / toy vocabs."""
    tok = card.get("tokenizer") or {}
    id_to_token = tok.get("idToToken") or {}
    # Keys may be stringified ints
    vocab: dict[str, int] = {}
    for k, v in id_to_token.items():
        token = str(v)
        vocab[token] = int(k)

    if not vocab:
        raise SystemExit("Char tokenizer card missing idToToken; cannot build tokenizer.json")

    bos_id = tok.get("bosId")
    eos_id = tok.get("eosId")
    pad_id = tok.get("padId")
    unk_token = "<unk>"
    if unk_token not in vocab:
        # Prefer an unused id; otherwise reuse pad/eos if present
        next_id = max(vocab.values()) + 1 if vocab else 0
        vocab[unk_token] = next_id

    added_tokens: list[dict[str, Any]] = []
    special_pairs = [
        ("<bos>", bos_id),
        ("<eos>", eos_id),
        ("<pad>", pad_id),
        (unk_token, vocab[unk_token]),
    ]
    seen_ids: set[int] = set()
    for content, tid in special_pairs:
        if tid is None:
            continue
        tid_i = int(tid)
        if tid_i in seen_ids:
            continue
        seen_ids.add(tid_i)
        # Ensure vocab has a string for this id
        token_str = next((t for t, i in vocab.items() if i == tid_i), content)
        if token_str not in vocab:
            vocab[token_str] = tid_i
        added_tokens.append(
            {
                "id": tid_i,
                "content": token_str,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"Regex": ""},
            "behavior": "Isolated",
            "invert": False,
        },
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": unk_token,
        },
    }


def write_tokenizer_files(card: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write tokenizer.json + tokenizer_config.json. Returns tokenizer meta for docs."""
    tok = card.get("tokenizer") or {}
    kind = tok.get("kind", "char")
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "char":
        tokenizer_json = build_char_tokenizer_json(card)
        (out_dir / "tokenizer.json").write_text(
            json.dumps(tokenizer_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        bos_id = tok.get("bosId")
        eos_id = tok.get("eosId")
        pad_id = tok.get("padId")
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": _block_size(card),
            "bos_token": None,
            "eos_token": None,
            "pad_token": None,
            "unk_token": "<unk>",
            "clean_up_tokenization_spaces": False,
        }
        # Map special ids back to token strings when present
        id_to_token = {int(k): str(v) for k, v in (tok.get("idToToken") or {}).items()}
        if bos_id is not None and int(bos_id) in id_to_token:
            tokenizer_config["bos_token"] = id_to_token[int(bos_id)]
        if eos_id is not None and int(eos_id) in id_to_token:
            tokenizer_config["eos_token"] = id_to_token[int(eos_id)]
        if pad_id is not None and int(pad_id) in id_to_token:
            tokenizer_config["pad_token"] = id_to_token[int(pad_id)]
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2), encoding="utf-8"
        )
        return {"kind": "char", "tokenizerFile": "tokenizer.json"}

    note = {
        "kind": "hf",
        "tokenizerFile": tok.get("tokenizerFile") or "tokenizer.json",
        "vocabSize": tok.get("vocabSize"),
        "note": (
            "Natural-language runs need the matching HuggingFace tokenizer.json "
            "from Browser Train static/tokenizer copied into this folder."
        ),
    }
    (out_dir / "tokenizer.note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    # Minimal stub so the folder is self-describing; AutoTokenizer will fail until user copies HF file
    stub_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": _block_size(card),
        "note": note["note"],
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(stub_config, indent=2), encoding="utf-8")
    return note


def decoder_gpt2_config(card: dict[str, Any]) -> dict[str, Any]:
    profile = card["exportProfile"]
    n_embd = int(card["embeddingSize"])
    n_layer = int(card["layers"]["decoder"])
    n_head = int(profile["nHeads"])
    vocab_size = int(card["vocabSize"])
    n_positions = _block_size(card)
    tok = card.get("tokenizer") or {}
    eos_id = tok.get("eosId")
    bos_id = tok.get("bosId")
    pad_id = tok.get("padId")
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_positions": n_positions,
        "n_ctx": n_positions,
        "max_position_embeddings": n_positions,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "transformers_version": "4.40.0",
        "browser_train_architecture": "decoder",
        "browser_train_dataset": card.get("dataset"),
    }


def encdec_package_config(card: dict[str, Any]) -> dict[str, Any]:
    """Honest config — not a fake T5/BART; AutoModelForSeq2SeqLM will not work."""
    profile = card["exportProfile"]
    return {
        "model_type": "browser_train_encoder_decoder",
        "architectures": ["BrowserTrainEncoderDecoder"],
        "browser_train_architecture": "encoder-decoder",
        "browser_train_dataset": card.get("dataset"),
        "vocab_size": int(card["vocabSize"]),
        "hidden_size": int(card["embeddingSize"]),
        "num_attention_heads": int(profile["nHeads"]),
        "num_encoder_layers": int(card["layers"]["encoder"]),
        "num_decoder_layers": int(card["layers"]["decoder"]),
        "max_source_positions": _block_size(card, "source"),
        "max_target_positions": _block_size(card, "target"),
        "note": (
            "Encoder-decoder AutoModel / pipeline is not supported in v1. "
            "Use the ORT package (ort/) with encodeDecode in browser-train-infer."
        ),
    }


def generation_config(card: dict[str, Any]) -> dict[str, Any]:
    tok = card.get("tokenizer") or {}
    return {
        "bos_token_id": tok.get("bosId"),
        "eos_token_id": tok.get("eosId"),
        "pad_token_id": tok.get("padId"),
        "max_length": _block_size(card),
        "do_sample": False,
    }


def export_decoder_onnx_for_tjs(
    module: Any,
    onnx_path: Path,
    opset: int = 17,
) -> None:
    import torch

    DecoderLMWithAttentionMask = _decoder_lm_with_attention_mask()
    wrapped = DecoderLMWithAttentionMask(module)
    wrapped.eval()
    seq = min(8, module.block_size)
    input_ids = torch.zeros((1, seq), dtype=torch.long)
    attention_mask = torch.ones((1, seq), dtype=torch.long)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
    )
    # Some Transformers.js GPT-2 loaders look for decoder_model_merged.onnx
    merged = onnx_path.parent / "decoder_model_merged.onnx"
    if merged.resolve() != onnx_path.resolve():
        shutil.copy2(onnx_path, merged)


def write_transformers_js_package(
    *,
    card: dict[str, Any],
    architecture: str,
    module: Any,
    out_dir: Path,
    opset: int = 17,
    ort_onnx_path: Path | None = None,
) -> None:
    """
    Write a Hub-style folder for Transformers.js.

    Decoder: gpt2 config + attention_mask-aligned ONNX (AutoModelForCausalLM path).
    EncDec: honest config + copied/custom ONNX; generation via ORT only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    (out_dir / "browser_train_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    write_tokenizer_files(card, out_dir)
    (out_dir / "generation_config.json").write_text(
        json.dumps(generation_config(card), indent=2), encoding="utf-8"
    )

    if architecture == "encoder-decoder":
        (out_dir / "config.json").write_text(
            json.dumps(encdec_package_config(card), indent=2), encoding="utf-8"
        )
        if ort_onnx_path and ort_onnx_path.is_file():
            shutil.copy2(ort_onnx_path, onnx_path)
        else:
            import torch

            from .models import EncoderDecoderLM

            if not isinstance(module, EncoderDecoderLM):
                raise SystemExit("EncDec Transformers.js package needs EncoderDecoderLM or ORT onnx")
            enc = torch.zeros((1, min(8, module.block_src)), dtype=torch.long)
            dec = torch.zeros((1, min(8, module.block_tgt)), dtype=torch.long)
            torch.onnx.export(
                module,
                (enc, dec),
                str(onnx_path),
                input_names=["encoder_input_ids", "decoder_input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "encoder_input_ids": {0: "batch", 1: "src_len"},
                    "decoder_input_ids": {0: "batch", 1: "tgt_len"},
                    "logits": {0: "batch", 1: "tgt_len"},
                },
                opset_version=opset,
            )
        (out_dir / "README.md").write_text(
            "\n".join(
                [
                    "# Browser Train encoder-decoder (Transformers.js package)",
                    "",
                    "This folder follows a Hub-style layout (config / tokenizer / onnx).",
                    "",
                    "**Generation via AutoModelForSeq2SeqLM is not supported in v1.**",
                    "Use the sibling `ort/` package with `browser-train-infer` `encodeDecode`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"==> transformers-js (EncDec layout only) → {out_dir}")
        return

    # decoder
    from .models import DecoderLM

    if not isinstance(module, DecoderLM):
        raise SystemExit("Decoder Transformers.js export requires DecoderLM")
    (out_dir / "config.json").write_text(
        json.dumps(decoder_gpt2_config(card), indent=2), encoding="utf-8"
    )
    print(f"==> exporting Transformers.js decoder ONNX → {onnx_path}")
    export_decoder_onnx_for_tjs(module, onnx_path, opset=opset)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Browser Train decoder (Transformers.js)",
                "",
                "Load locally with `@huggingface/transformers`:",
                "",
                "```js",
                "import { env, AutoTokenizer, AutoModelForCausalLM } from '@huggingface/transformers';",
                "env.allowLocalModels = true;",
                "env.allowRemoteModels = false;",
                "env.localModelPath = '/'; // parent of this folder",
                "const tokenizer = await AutoTokenizer.from_pretrained('transformers-js');",
                "const model = await AutoModelForCausalLM.from_pretrained('transformers-js', { dtype: 'fp32' });",
                "```",
                "",
                "See `examples/browser-train-infer-tjs` for a minimal demo.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"==> transformers-js (decoder / gpt2) → {out_dir}")
