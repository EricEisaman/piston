"""Convert Browser Train inference packages to ONNX (+ optional Transformers.js layout)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .validate import assert_exportable


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


def _checkpoint_stem(checkpoint: Path) -> str:
    name = checkpoint.name
    if name.endswith(".inference.safetensors"):
        return name[: -len(".inference.safetensors")]
    if name.endswith(".safetensors"):
        return name[: -len(".safetensors")]
    return checkpoint.stem


def _guess_model_json(checkpoint: Path) -> Path | None:
    guess = checkpoint.parent / f"{_checkpoint_stem(checkpoint)}.model.json"
    return guess if guess.is_file() else None


def _guess_tokenizer_json(checkpoint: Path) -> Path | None:
    """Browser Train downloads `{run}.tokenizer.json` next to the inference package."""
    guess = checkpoint.parent / f"{_checkpoint_stem(checkpoint)}.tokenizer.json"
    if guess.is_file():
        return guess
    plain = checkpoint.parent / "tokenizer.json"
    return plain if plain.is_file() else None


def _sibling_tokenizer_config(tokenizer_src: Path) -> Path | None:
    candidates: list[Path] = [tokenizer_src.with_name("tokenizer_config.json")]
    name = tokenizer_src.name
    if name.endswith(".tokenizer.json"):
        stem = name[: -len(".tokenizer.json")]
        candidates.insert(0, tokenizer_src.parent / f"{stem}.tokenizer_config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _install_hf_tokenizer_files(
    out_dir: Path,
    tokenizer_src: Path | None,
    *,
    vocab_size: Any = None,
) -> dict[str, Any]:
    """Copy HF tokenizer into out_dir when available; otherwise write tokenizer.note.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer_src is not None and tokenizer_src.is_file():
        dest = out_dir / "tokenizer.json"
        shutil.copy2(tokenizer_src, dest)
        config_src = _sibling_tokenizer_config(tokenizer_src)
        if config_src is not None:
            shutil.copy2(config_src, out_dir / "tokenizer_config.json")
        print(f"==> installed HF tokenizer → {dest}")
        return {
            "kind": "hf",
            "tokenizerFile": "tokenizer.json",
            "vocabSize": vocab_size,
            "installedFrom": str(tokenizer_src),
        }

    note = {
        "kind": "hf",
        "tokenizerFile": "tokenizer.json",
        "vocabSize": vocab_size,
        "note": (
            "Copy the matching HF tokenizer.json next to the checkpoint "
            "(Browser Train purple download emits {run}.tokenizer.json) or pass --tokenizer."
        ),
    }
    (out_dir / "tokenizer.note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    return note


def _write_ort_tokenizer(
    card: dict[str, Any],
    out_dir: Path,
    tokenizer_src: Path | None = None,
) -> dict[str, Any]:
    tok = card.get("tokenizer") or {}
    kind = tok.get("kind", "char")
    if kind == "char":
        id_to_token = tok.get("idToToken") or {}
        vocab = {v: int(k) for k, v in id_to_token.items()}
        special = {
            "bosId": tok.get("bosId"),
            "eosId": tok.get("eosId"),
            "padId": tok.get("padId"),
            "maskId": tok.get("maskId"),
            "maskToken": next(
                (str(v) for k, v in id_to_token.items() if int(k) == int(tok["maskId"])),
                "<mask>",
            )
            if tok.get("maskId") is not None
            else None,
        }
        (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        (out_dir / "special_tokens.json").write_text(json.dumps(special, indent=2), encoding="utf-8")
        return {"kind": "char", "vocabFile": "vocab.json", "specialTokensFile": "special_tokens.json"}

    return _install_hf_tokenizer_files(
        out_dir,
        tokenizer_src,
        vocab_size=tok.get("vocabSize"),
    )


def parse_targets(raw: str) -> set[str]:
    text = raw.strip().lower()
    if text in ("both", "all"):
        return {"ort", "transformers-js"}
    parts = {p.strip() for p in text.split(",") if p.strip()}
    allowed = {"ort", "transformers-js"}
    unknown = parts - allowed
    if unknown:
        raise SystemExit(
            f"Unknown --targets {sorted(unknown)}; use ort, transformers-js, or both"
        )
    if not parts:
        raise SystemExit("--targets must include ort and/or transformers-js")
    return parts


def _export_ort_onnx(
    architecture: str,
    module: Any,
    onnx_path: Path,
    opset: int,
) -> list[dict[str, str]]:
    import torch

    from .models import DecoderLM, EncoderDecoderLM, EncoderLM

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
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
            dynamo=False,
        )
        return [
            {"name": "encoder_input_ids", "dtype": "int64"},
            {"name": "decoder_input_ids", "dtype": "int64"},
        ]

    if architecture == "encoder":
        assert isinstance(module, EncoderLM)
        seq = min(8, module.block_size)
        ids = torch.zeros((1, seq), dtype=torch.long)
        mask = torch.ones((1, seq), dtype=torch.long)
        types = torch.zeros((1, seq), dtype=torch.long)
        torch.onnx.export(
            module,
            (ids, mask, types),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
            dynamo=False,
        )
        return [
            {"name": "input_ids", "dtype": "int64"},
            {"name": "attention_mask", "dtype": "int64"},
            {"name": "token_type_ids", "dtype": "int64"},
        ]

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
        dynamo=False,
    )
    return [{"name": "input_ids", "dtype": "int64"}]


def write_ort_package(
    *,
    card: dict[str, Any],
    architecture: str,
    module: Any,
    out_dir: Path,
    opset: int,
    tokenizer_src: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    print(f"==> exporting ORT {onnx_path}")
    inputs = _export_ort_onnx(architecture, module, onnx_path, opset)
    tok_meta = _write_ort_tokenizer(card, out_dir, tokenizer_src=tokenizer_src)
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
    print(f"==> wrote ORT package {out_dir}")
    return onnx_path


def convert(
    checkpoint: Path,
    out_dir: Path,
    model_json: Path | None = None,
    opset: int = 17,
    targets: set[str] | None = None,
    tokenizer: Path | None = None,
) -> None:
    from .transformers_js import write_transformers_js_package
    from .weights import (
        build_decoder_from_card,
        build_encdec_from_card,
        build_encoder_from_card,
        load_into_module,
        load_tensors,
    )

    targets = targets or {"ort", "transformers-js"}
    out_dir.mkdir(parents=True, exist_ok=True)
    card = _load_model_card(checkpoint, model_json)
    assert_exportable(card)
    tokenizer_src = tokenizer if tokenizer and tokenizer.is_file() else _guess_tokenizer_json(checkpoint)
    if tokenizer_src:
        print(f"==> HF tokenizer source {tokenizer_src}")
    elif (card.get("tokenizer") or {}).get("kind") == "hf":
        print(
            "==> warning: no {run}.tokenizer.json found beside checkpoint; "
            "Transformers.js / HF apps need --tokenizer or the purple-download sidecar"
        )

    architecture = card.get("architecture") or card["config"]["model"]["topology"]
    if architecture not in ("decoder", "encoder-decoder", "encoder"):
        raise SystemExit(f'Unsupported architecture "{architecture}"')
    print(f"==> architecture={architecture} targets={sorted(targets)}")

    tensors = load_tensors(str(checkpoint))
    print(f"==> {len(tensors)} model tensors")

    if architecture == "encoder-decoder":
        module: Any = build_encdec_from_card(card)
    elif architecture == "encoder":
        module = build_encoder_from_card(card)
    else:
        module = build_decoder_from_card(card)
    module.eval()
    load_into_module(module, tensors, architecture)

    want_ort = "ort" in targets
    want_tjs = "transformers-js" in targets

    # Layout:
    # - ort only → flat out/ (back-compat)
    # - transformers-js only → HF layout at out/
    # - both → out/ort/ and out/transformers-js/
    if want_ort and want_tjs:
        ort_dir = out_dir / "ort"
        tjs_dir = out_dir / "transformers-js"
    elif want_ort:
        ort_dir = out_dir
        tjs_dir = None
    else:
        ort_dir = None
        tjs_dir = out_dir

    ort_onnx: Path | None = None
    if want_ort and ort_dir is not None:
        ort_onnx = write_ort_package(
            card=card,
            architecture=architecture,
            module=module,
            out_dir=ort_dir,
            opset=opset,
            tokenizer_src=tokenizer_src,
        )

    if want_tjs and tjs_dir is not None:
        write_transformers_js_package(
            card=card,
            architecture=architecture,
            module=module,
            out_dir=tjs_dir,
            opset=opset,
            ort_onnx_path=ort_onnx,
            tokenizer_src=tokenizer_src,
        )

    print(f"==> done → {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Browser Train → ONNX / Transformers.js converter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    convert_p = sub.add_parser("convert", help="Convert inference safetensors to ONNX packages")
    convert_p.add_argument("checkpoint", type=Path, help="*.inference.safetensors (or training ckpt)")
    convert_p.add_argument("-o", "--out", type=Path, required=True, help="Output directory")
    convert_p.add_argument(
        "--model-json",
        type=Path,
        default=None,
        help="Sidecar *.model.json from Browser Train download",
    )
    convert_p.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="HF tokenizer.json (defaults to {run}.tokenizer.json beside the checkpoint)",
    )
    convert_p.add_argument("--opset", type=int, default=17)
    convert_p.add_argument(
        "--targets",
        type=str,
        default="both",
        help="Comma list: ort, transformers-js, or both (default: both)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "convert":
        model_json = args.model_json or _guess_model_json(args.checkpoint)
        convert(
            args.checkpoint,
            args.out,
            model_json=model_json,
            opset=args.opset,
            targets=parse_targets(args.targets),
            tokenizer=args.tokenizer,
        )
    else:
        parser.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
