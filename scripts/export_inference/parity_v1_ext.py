"""
v1-ext parity checks (synthetic weights — no trained checkpoint required).

1. EncDec: monolithic forward (ORT encodeDecode graph) vs encode+decode (TJS Seq2Seq graphs)
2. Encoder MLM: ORT model.onnx vs TJS onnx/model.onnx argmax at mask positions

Run:
  PYTHONPATH=scripts python -m export_inference.parity_v1_ext
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _toy_mlp_config() -> dict[str, Any]:
    return {
        "model": {
            "layers": 2,
            "transformer": {
                "mlp": {
                    "hiddenExpansionFactor": 4,
                    "activation": "gelu",
                }
            },
        }
    }


def _export_profile() -> dict[str, Any]:
    return {
        "attentionGating": False,
        "attentionPresent": True,
        "gqa": False,
        "mlpGated": False,
        "mlpPresent": True,
        "nHeads": 2,
        "nKvHeads": 2,
        "positionalEncoding": "learned",
        "qkNorm": False,
        "sinks": False,
        "softcapAttention": False,
        "softcapLogits": False,
    }


def _encdec_card() -> dict[str, Any]:
    return {
        "architecture": "encoder-decoder",
        "vocabSize": 32,
        "embeddingSize": 32,
        "blockSize": {"source": 16, "target": 16},
        "layers": {"encoder": 1, "decoder": 1},
        "exportProfile": _export_profile(),
        "config": _toy_mlp_config(),
        "dataset": "sort",
        "tokenizer": {
            "kind": "char",
            "idToToken": {str(i): chr(65 + i) if i < 26 else f"<{i}>" for i in range(32)},
            "bosId": 0,
            "eosId": 1,
            "padId": 2,
            "maskId": None,
        },
    }


def _encoder_card() -> dict[str, Any]:
    id_to_token = {str(i): t for i, t in enumerate(["(", ")", "[", "]", "<eos>", "<mask>"])}
    return {
        "architecture": "encoder",
        "vocabSize": 6,
        "embeddingSize": 32,
        "blockSize": 16,
        "layers": {"encoder": 2, "decoder": 0},
        "exportProfile": _export_profile(),
        "config": _toy_mlp_config(),
        "dataset": "dyck",
        "tokenizer": {
            "kind": "char",
            "idToToken": id_to_token,
            "bosId": None,
            "eosId": 4,
            "padId": None,
            "maskId": 5,
        },
    }


def _check_encdec_torch() -> None:
    from .models import EncoderDecoderLM

    torch.manual_seed(0)
    m = EncoderDecoderLM(
        vocab_size=32,
        n_embed=32,
        n_heads=2,
        n_encoder=1,
        n_decoder=1,
        block_src=16,
        block_tgt=16,
        expansion=4.0,
        gated=False,
        activation="gelu",
        use_pos_emb=True,
    )
    m.eval()
    enc = torch.randint(0, 32, (1, 5))
    dec = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        mono = m(enc, dec)
        split = m.decode(dec, m.encode(enc))
    if not torch.allclose(mono, split, atol=1e-5, rtol=1e-5):
        raise SystemExit("EncDec parity FAIL: monolithic != encode+decode")
    print("OK EncDec torch: monolithic matches encode+decode")


def _check_encdec_onnx() -> None:
    from .convert import write_ort_package
    from .transformers_js import write_transformers_js_package
    from .weights import build_encdec_from_card

    card = _encdec_card()
    module = build_encdec_from_card(card)
    module.eval()
    torch.manual_seed(1)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.02)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ort_dir = root / "ort"
        tjs_dir = root / "tjs"
        write_ort_package(card=card, architecture="encoder-decoder", module=module, out_dir=ort_dir, opset=17)
        write_transformers_js_package(
            card=card, architecture="encoder-decoder", module=module, out_dir=tjs_dir, opset=17
        )

        try:
            import onnxruntime as ort
        except ImportError:
            print("SKIP EncDec ONNX (install onnxruntime for graph parity)")
            return

        enc = np.array([[3, 2, 1, 0, 4]], dtype=np.int64)
        dec = np.array([[0, 1, 2]], dtype=np.int64)
        sess_ort = ort.InferenceSession(str(ort_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        logits_ort = sess_ort.run(None, {"encoder_input_ids": enc, "decoder_input_ids": dec})[0]

        sess_enc = ort.InferenceSession(
            str(tjs_dir / "onnx" / "encoder_model.onnx"), providers=["CPUExecutionProvider"]
        )
        attn = np.ones_like(enc)
        hidden = sess_enc.run(None, {"input_ids": enc, "attention_mask": attn})[0]
        sess_dec = ort.InferenceSession(
            str(tjs_dir / "onnx" / "decoder_model.onnx"), providers=["CPUExecutionProvider"]
        )
        logits_tjs = sess_dec.run(
            None,
            {
                "input_ids": dec,
                "encoder_hidden_states": hidden,
                "encoder_attention_mask": attn,
            },
        )[0]

        if not np.allclose(logits_ort, logits_tjs, atol=1e-4, rtol=1e-4):
            raise SystemExit(
                f"EncDec ONNX parity FAIL: max abs diff={np.max(np.abs(logits_ort - logits_tjs))}"
            )
        arg_ort = logits_ort.argmax(axis=-1)
        arg_tjs = logits_tjs.argmax(axis=-1)
        if not np.array_equal(arg_ort, arg_tjs):
            raise SystemExit(f"EncDec ONNX argmax FAIL: {arg_ort} vs {arg_tjs}")
        print("OK EncDec ONNX: ORT monolith matches TJS encoder+decoder (logits + argmax)")


def _check_encoder_onnx() -> None:
    from .convert import write_ort_package
    from .transformers_js import write_transformers_js_package
    from .weights import build_encoder_from_card

    card = _encoder_card()
    module = build_encoder_from_card(card)
    module.eval()
    torch.manual_seed(2)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.02)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ort_dir = root / "ort"
        tjs_dir = root / "tjs"
        write_ort_package(card=card, architecture="encoder", module=module, out_dir=ort_dir, opset=17)
        write_transformers_js_package(
            card=card, architecture="encoder", module=module, out_dir=tjs_dir, opset=17
        )

        try:
            import onnxruntime as ort
        except ImportError:
            print("SKIP encoder ONNX (install onnxruntime for graph parity)")
            return

        # "( <mask> )" → ids 0, 5, 1
        ids = np.array([[0, 5, 1]], dtype=np.int64)
        mask = np.ones_like(ids)
        types = np.zeros_like(ids)
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}

        sess_ort = ort.InferenceSession(str(ort_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        sess_tjs = ort.InferenceSession(
            str(tjs_dir / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        logits_ort = sess_ort.run(None, feeds)[0]
        logits_tjs = sess_tjs.run(None, feeds)[0]
        if not np.allclose(logits_ort, logits_tjs, atol=1e-4, rtol=1e-4):
            raise SystemExit(
                f"Encoder ONNX parity FAIL: max abs diff={np.max(np.abs(logits_ort - logits_tjs))}"
            )
        mask_pos = 1
        a_ort = int(logits_ort[0, mask_pos].argmax())
        a_tjs = int(logits_tjs[0, mask_pos].argmax())
        if a_ort != a_tjs:
            raise SystemExit(f"Encoder mask argmax FAIL: {a_ort} vs {a_tjs}")
        print("OK Encoder ONNX: ORT matches TJS MaskedLM (logits + mask argmax)")


def main() -> None:
    print("==> v1-ext parity (synthetic)")
    _check_encdec_torch()
    _check_encdec_onnx()
    _check_encoder_onnx()
    print("==> all parity checks passed")


if __name__ == "__main__":
    main()
