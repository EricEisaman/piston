"""Convert Browser Train inference packages to ONNX + tokenizer/manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .models import DecoderLM, EncoderDecoderLM
from .validate import assert_exportable
from .weights import (
    build_decoder_from_card,
    build_encdec_from_card,
    load_into_module,
    load_tensors,
)


def _load_model_card(checkpoint: Path, model_json: Path | None) -> dict[str, Any]:
    if model_json and model_json.is_file():
        return json.loads(model_json.read_text(encoding="utf-8"))

    # Fallback: try to read piston_extra from safetensors metadata
    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint), framework="pt") as f:
            meta = f.metadata() or {}
            raw = meta.get("piston_extra")
            if raw:
                extra = json.loads(raw)
                if isinstance(extra, dict) and "model" in extra:
                    return extra["model"]
                if isinstance(extra, dict) and extra.get("format") == "browser-train-inference-v1":
                    return extra.get("model") or extra
    except Exception as exc:  # noqa: BLE001
        print(f"  note: could not read embedded metadata ({exc})")

    raise SystemExit(
        f"Missing model card. Pass --model-json path/to/*.model.json next to {checkpoint.name}"
    )


def _write_tokenizer(card: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    tok = card.get("tokenizer") or {}
    kind = tok.get("kind", "char")
    if kind == "char":
        id_to_token = tok.get("idToToken") or {}
        vocab = {v: int(k) for k, v in id_to_token.items()}
        special = {
            "bosId": tok.get("bosId"),
            "eosId": tok.get("eosId"),
            "padId": tok.get("padId"),
        }
        (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        (out_dir / "special_tokens.json").write_text(json.dumps(special, indent=2), encoding="utf-8")
        return {"kind": "char", "vocabFile": "vocab.json", "specialTokensFile": "special_tokens.json"}

    note = {
        "kind": "hf",
        "tokenizerFile": tok.get("tokenizerFile") or "tokenizer.json",
        "vocabSize": tok.get("vocabSize"),
        "note": "Copy the matching HF tokenizer.json from Browser Train static/tokenizer into this folder.",
    }
    (out_dir / "tokenizer.note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    return note


def convert(
    checkpoint: Path,
    out_dir: Path,
    model_json: Path | None = None,
    opset: int = 17,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    card = _load_model_card(checkpoint, model_json)
    assert_exportable(card)

    architecture = card.get("architecture") or card["config"]["model"]["topology"]
    if architecture == "encoder":
        architecture = "decoder"
    print(f"==> architecture={architecture}")

    tensors = load_tensors(str(checkpoint))
    print(f"==> {len(tensors)} model tensors")

    if architecture == "encoder-decoder":
        module: torch.nn.Module = build_encdec_from_card(card)
    else:
        module = build_decoder_from_card(card)
    module.eval()
    load_into_module(module, tensors, architecture)

    onnx_path = out_dir / "model.onnx"
    print(f"==> exporting {onnx_path}")

    if architecture == "encoder-decoder":
        assert isinstance(module, EncoderDecoderLM)
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
        inputs = [
            {"name": "encoder_input_ids", "dtype": "int64"},
            {"name": "decoder_input_ids", "dtype": "int64"},
        ]
    else:
        assert isinstance(module, DecoderLM)
        ids = torch.zeros((1, min(8, module.block_size)), dtype=torch.long)
        torch.onnx.export(
            module,
            ids,
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
        )
        inputs = [{"name": "input_ids", "dtype": "int64"}]

    tok_meta = _write_tokenizer(card, out_dir)
    (out_dir / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    manifest = {
        "architecture": architecture,
        "onnx": "model.onnx",
        "inputs": inputs,
        "outputs": [{"name": "logits", "dtype": "float32"}],
        "vocabSize": card["vocabSize"],
        "embeddingSize": card["embeddingSize"],
        "tokenizer": tok_meta,
        "dataset": card.get("dataset"),
    }
    (out_dir / "ort-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"==> wrote {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Browser Train → ONNX converter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    convert_p = sub.add_parser("convert", help="Convert inference safetensors to ONNX")
    convert_p.add_argument("checkpoint", type=Path, help="*.inference.safetensors (or training ckpt)")
    convert_p.add_argument("-o", "--out", type=Path, required=True, help="Output directory")
    convert_p.add_argument(
        "--model-json",
        type=Path,
        default=None,
        help="Sidecar *.model.json from Browser Train download",
    )
    convert_p.add_argument("--opset", type=int, default=17)

    args = parser.parse_args(argv)
    if args.cmd == "convert":
        model_json = args.model_json
        if model_json is None:
            guess = args.checkpoint.with_suffix("").with_suffix(".model.json")
            # yearning-emu-0.inference.safetensors → yearning-emu-0.model.json
            stem = args.checkpoint.name
            if stem.endswith(".inference.safetensors"):
                guess = args.checkpoint.parent / (stem[: -len(".inference.safetensors")] + ".model.json")
            elif stem.endswith(".safetensors"):
                guess = args.checkpoint.parent / (stem[: -len(".safetensors")] + ".model.json")
            if guess.is_file():
                model_json = guess
        convert(args.checkpoint, args.out, model_json=model_json, opset=args.opset)
    else:
        parser.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
