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

    mask_id = tok.get("maskId")
    if mask_id is not None:
        mask_id_i = int(mask_id)
        if "<mask>" not in vocab:
            # Prefer the string already mapped for this id
            existing = next((t for t, i in vocab.items() if i == mask_id_i), "<mask>")
            vocab[existing if existing else "<mask>"] = mask_id_i
            if existing != "<mask>" and "<mask>" not in vocab:
                vocab["<mask>"] = mask_id_i
        elif vocab["<mask>"] != mask_id_i:
            vocab["<mask>"] = mask_id_i

    added_tokens: list[dict[str, Any]] = []
    special_pairs = [
        ("<bos>", bos_id),
        ("<eos>", eos_id),
        ("<pad>", pad_id),
        ("<mask>", mask_id),
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


def write_tokenizer_files(
    card: dict[str, Any],
    out_dir: Path,
    tokenizer_src: Path | None = None,
) -> dict[str, Any]:
    """Write tokenizer.json + tokenizer_config.json. Returns tokenizer meta for docs."""
    from .convert import _install_hf_tokenizer_files

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
            "mask_token": None,
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
        mask_id = tok.get("maskId")
        if mask_id is not None and int(mask_id) in id_to_token:
            tokenizer_config["mask_token"] = id_to_token[int(mask_id)]
        elif "<mask>" in {str(v) for v in id_to_token.values()}:
            tokenizer_config["mask_token"] = "<mask>"
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2), encoding="utf-8"
        )
        return {"kind": "char", "tokenizerFile": "tokenizer.json"}

    meta = _install_hf_tokenizer_files(
        out_dir,
        tokenizer_src,
        vocab_size=tok.get("vocabSize"),
    )
    # Ensure a tokenizer_config.json exists for AutoTokenizer even when only tokenizer.json was copied.
    config_path = out_dir / "tokenizer_config.json"
    if not config_path.is_file():
        stub_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": _block_size(card),
            "clean_up_tokenization_spaces": False,
        }
        config_path.write_text(json.dumps(stub_config, indent=2), encoding="utf-8")
    return meta


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


def encdec_bart_config(card: dict[str, Any]) -> dict[str, Any]:
    """BART-compatible config for AutoModelForSeq2SeqLM (v1-ext)."""
    profile = card["exportProfile"]
    tok = card.get("tokenizer") or {}
    n_embd = int(card["embeddingSize"])
    n_head = int(profile["nHeads"])
    bos_id = tok.get("bosId")
    eos_id = tok.get("eosId")
    pad_id = tok.get("padId")
    decoder_start = bos_id if bos_id is not None else eos_id
    return {
        "architectures": ["BartForConditionalGeneration"],
        "model_type": "bart",
        "vocab_size": int(card["vocabSize"]),
        "d_model": n_embd,
        "encoder_layers": int(card["layers"]["encoder"]),
        "decoder_layers": int(card["layers"]["decoder"]),
        "encoder_attention_heads": n_head,
        "decoder_attention_heads": n_head,
        "encoder_ffn_dim": int(n_embd * 4),
        "decoder_ffn_dim": int(n_embd * 4),
        "max_position_embeddings": max(_block_size(card, "source"), _block_size(card, "target")),
        "activation_function": "gelu",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "init_std": 0.02,
        "classifier_dropout": 0.0,
        "scale_embedding": False,
        "use_cache": True,
        "num_hidden_layers": int(card["layers"]["encoder"]),
        "pad_token_id": pad_id,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "decoder_start_token_id": decoder_start,
        "forced_eos_token_id": eos_id,
        "transformers_version": "4.40.0",
        "browser_train_architecture": "encoder-decoder",
        "browser_train_dataset": card.get("dataset"),
    }


def encoder_bert_config(card: dict[str, Any]) -> dict[str, Any]:
    """BERT-compatible config for AutoModelForMaskedLM (v1-ext Dyck)."""
    profile = card["exportProfile"]
    tok = card.get("tokenizer") or {}
    n_embd = int(card["embeddingSize"])
    n_layers = int(card["layers"].get("encoder") or card["layers"].get("decoder") or 2)
    return {
        "architectures": ["BertForMaskedLM"],
        "model_type": "bert",
        "vocab_size": int(card["vocabSize"]),
        "hidden_size": n_embd,
        "num_hidden_layers": n_layers,
        "num_attention_heads": int(profile["nHeads"]),
        "intermediate_size": int(n_embd * 4),
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": _block_size(card, "source"),
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-5,
        "pad_token_id": tok.get("padId"),
        "bos_token_id": tok.get("bosId"),
        "eos_token_id": tok.get("eosId"),
        "mask_token_id": tok.get("maskId"),
        "transformers_version": "4.40.0",
        "browser_train_architecture": "encoder",
        "browser_train_dataset": card.get("dataset"),
    }


def generation_config(card: dict[str, Any]) -> dict[str, Any]:
    tok = card.get("tokenizer") or {}
    bos_id = tok.get("bosId")
    eos_id = tok.get("eosId")
    return {
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": tok.get("padId"),
        "decoder_start_token_id": bos_id if bos_id is not None else eos_id,
        "max_length": _block_size(card),
        "do_sample": False,
    }


def export_encdec_onnx_for_tjs(module: Any, onnx_dir: Path, opset: int = 17) -> None:
    """Write encoder_model.onnx + decoder_model.onnx (+ merged copy) for TJS Seq2Seq."""
    import torch

    from .models import EncDecDecoderExport, EncDecEncoderExport, EncoderDecoderLM

    if not isinstance(module, EncoderDecoderLM):
        raise SystemExit("EncDec Transformers.js export requires EncoderDecoderLM")

    onnx_dir.mkdir(parents=True, exist_ok=True)
    enc_path = onnx_dir / "encoder_model.onnx"
    dec_path = onnx_dir / "decoder_model.onnx"

    enc_wrap = EncDecEncoderExport(module)
    dec_wrap = EncDecDecoderExport(module)
    enc_wrap.eval()
    dec_wrap.eval()

    src_len = min(8, module.block_src)
    tgt_len = min(8, module.block_tgt)
    input_ids = torch.zeros((1, src_len), dtype=torch.long)
    attention_mask = torch.ones((1, src_len), dtype=torch.long)
    enc_hidden = torch.zeros((1, src_len, module.encWordEmbedding.embedding_dim), dtype=torch.float32)
    dec_ids = torch.zeros((1, tgt_len), dtype=torch.long)

    print(f"==> exporting TJS EncDec encoder → {enc_path}")
    torch.onnx.export(
        enc_wrap,
        (input_ids, attention_mask),
        str(enc_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
            "last_hidden_state": {0: "batch", 1: "encoder_sequence"},
        },
        opset_version=opset,
        dynamo=False,
    )

    print(f"==> exporting TJS EncDec decoder → {dec_path}")
    torch.onnx.export(
        dec_wrap,
        (dec_ids, enc_hidden, attention_mask),
        str(dec_path),
        input_names=["input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "decoder_sequence"},
            "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
            "encoder_attention_mask": {0: "batch", 1: "encoder_sequence"},
            "logits": {0: "batch", 1: "decoder_sequence"},
        },
        opset_version=opset,
        dynamo=False,
    )
    # TJS file probe often looks for decoder_model_merged.onnx
    shutil.copy2(dec_path, onnx_dir / "decoder_model_merged.onnx")


def export_encoder_onnx_for_tjs(module: Any, onnx_path: Path, opset: int = 17) -> None:
    import torch

    from .models import EncoderLM

    if not isinstance(module, EncoderLM):
        raise SystemExit("Encoder Transformers.js export requires EncoderLM")
    module.eval()
    seq = min(8, module.block_size)
    input_ids = torch.zeros((1, seq), dtype=torch.long)
    attention_mask = torch.ones((1, seq), dtype=torch.long)
    token_type_ids = torch.zeros((1, seq), dtype=torch.long)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        dynamo=False,
    )


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
        dynamo=False,
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
    tokenizer_src: Path | None = None,
) -> None:
    """
    Write a Hub-style folder for Transformers.js.

    Decoder: gpt2 + CausalLM.
    EncDec: BART + Seq2SeqLM (v1-ext).
    Encoder: BERT + MaskedLM (v1-ext Dyck).
    """
    _ = ort_onnx_path  # ORT monolith kept separate; TJS EncDec uses split graphs
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    (out_dir / "browser_train_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    write_tokenizer_files(card, out_dir, tokenizer_src=tokenizer_src)
    (out_dir / "generation_config.json").write_text(
        json.dumps(generation_config(card), indent=2), encoding="utf-8"
    )

    if architecture == "encoder-decoder":
        (out_dir / "config.json").write_text(
            json.dumps(encdec_bart_config(card), indent=2), encoding="utf-8"
        )
        export_encdec_onnx_for_tjs(module, onnx_dir, opset=opset)
        (out_dir / "README.md").write_text(
            "\n".join(
                [
                    "# Browser Train encoder-decoder (Transformers.js / v1-ext)",
                    "",
                    "Load with `AutoModelForSeq2SeqLM` + `AutoTokenizer` (BART-compatible packaging).",
                    "",
                    "```js",
                    "import { AutoModelForSeq2SeqLM, AutoTokenizer } from '@huggingface/transformers';",
                    "const tokenizer = await AutoTokenizer.from_pretrained('transformers-js');",
                    "const model = await AutoModelForSeq2SeqLM.from_pretrained('transformers-js', { dtype: 'fp32' });",
                    "const out = await model.generate(await tokenizer('CBA:', { return_tensors: 'pt' }));",
                    "```",
                    "",
                    "ORT fallback: sibling `ort/` package + `encodeDecode` in browser-train-infer.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"==> transformers-js (EncDec Seq2Seq / BART) → {out_dir}")
        return

    if architecture == "encoder":
        from .models import EncoderLM

        if not isinstance(module, EncoderLM):
            raise SystemExit("Encoder Transformers.js export requires EncoderLM")
        (out_dir / "config.json").write_text(
            json.dumps(encoder_bert_config(card), indent=2), encoding="utf-8"
        )
        print(f"==> exporting Transformers.js encoder MLM ONNX → {onnx_path}")
        export_encoder_onnx_for_tjs(module, onnx_path, opset=opset)
        (out_dir / "README.md").write_text(
            "\n".join(
                [
                    "# Browser Train encoder / MLM (Transformers.js / v1-ext)",
                    "",
                    "Load with `AutoModelForMaskedLM` + `AutoTokenizer` (BERT-compatible packaging).",
                    "",
                    "```js",
                    "import { AutoModelForMaskedLM, AutoTokenizer } from '@huggingface/transformers';",
                    "const tokenizer = await AutoTokenizer.from_pretrained('transformers-js');",
                    "const model = await AutoModelForMaskedLM.from_pretrained('transformers-js', { dtype: 'fp32' });",
                    "```",
                    "",
                    "ORT fallback: sibling `ort/` package + `encodeMasked` in browser-train-infer.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"==> transformers-js (encoder MaskedLM / BERT) → {out_dir}")
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
