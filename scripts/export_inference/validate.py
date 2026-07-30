"""Validate Browser Train model cards for ONNX export v1."""

from __future__ import annotations

from typing import Any


def get_blockers(model_card: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    config = model_card.get("config") or {}
    model = config.get("model") or {}
    family = model.get("family")
    topology = model.get("topology")
    profile = model_card.get("exportProfile") or {}

    if family != "transformer":
        blockers.append(f'model.family must be "transformer" (got "{family}")')
    if topology not in ("decoder", "encoder-decoder"):
        blockers.append(
            f'model.topology must be "decoder" or "encoder-decoder" (got "{topology}")'
        )
    if not profile.get("attentionPresent", True):
        blockers.append("transformer attention must be enabled")
    if profile.get("gqa"):
        blockers.append("disable grouped-query attention (GQA)")
    if profile.get("attentionGating"):
        blockers.append("disable attention gating")
    if profile.get("sinks"):
        blockers.append("disable attention sinks")
    if profile.get("softcapAttention") or profile.get("softcapLogits"):
        blockers.append("disable attention/logits softcap")
    pe = profile.get("positionalEncoding")
    if pe in ("rope", "alibi", "other"):
        blockers.append(f'positional encoding must be learned/sinusoidal/off (got "{pe}")')
    if not profile.get("mlpPresent", True):
        blockers.append("transformer MLP must be enabled")
    return blockers


def assert_exportable(model_card: dict[str, Any]) -> None:
    blockers = get_blockers(model_card)
    if blockers:
        raise SystemExit(
            "Checkpoint is not ONNX-exportable:\n- "
            + "\n- ".join(blockers)
            + '\n\nApply the "onnx-export-friendly" preset and retrain.'
        )
