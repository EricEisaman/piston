"""Load Browser Train safetensors into reference PyTorch modules."""

from __future__ import annotations

import re
from typing import Any

import torch
from safetensors.torch import load_file

from .models import DecoderLM, EncoderDecoderLM, EncoderLM


def _tensor_to_torch(t: torch.Tensor) -> torch.Tensor:
	return t.detach().cpu().float()


def load_tensors(path: str) -> dict[str, torch.Tensor]:
	raw = load_file(path)
	return {k: _tensor_to_torch(v) for k, v in raw.items() if not k.startswith("optimizer.state")}


def _strip_prefixes(key: str) -> str:
	k = key
	k = k.replace("decoder.dict.", "decoder.")
	k = k.replace("encoder.dict.", "encoder.")
	return k


def build_decoder_from_card(card: dict[str, Any]) -> DecoderLM:
	profile = card["exportProfile"]
	config = card["config"]
	mlp = config["model"]["transformer"]["mlp"]
	block = card["blockSize"]
	if isinstance(block, dict):
		block_size = int(block.get("target") or block.get("source") or 64)
	else:
		block_size = int(block)
	use_pos = profile.get("positionalEncoding") in ("learned", "sinusoidal")
	return DecoderLM(
		vocab_size=int(card["vocabSize"]),
		n_embed=int(card["embeddingSize"]),
		n_heads=int(profile["nHeads"]),
		n_layers=int(card["layers"]["decoder"]),
		block_size=max(block_size, 1),
		expansion=float(mlp.get("hiddenExpansionFactor", 4)),
		gated=bool(profile.get("mlpGated")),
		activation=str(mlp.get("activation", "gelu")),
		use_pos_emb=use_pos and profile.get("positionalEncoding") == "learned",
	)


def build_encdec_from_card(card: dict[str, Any]) -> EncoderDecoderLM:
	profile = card["exportProfile"]
	config = card["config"]
	mlp = config["model"]["transformer"]["mlp"]
	block = card["blockSize"]
	if isinstance(block, dict):
		block_src = int(block.get("source", 32))
		block_tgt = int(block.get("target", 32))
	else:
		block_src = block_tgt = int(block)
	return EncoderDecoderLM(
		vocab_size=int(card["vocabSize"]),
		n_embed=int(card["embeddingSize"]),
		n_heads=int(profile["nHeads"]),
		n_encoder=int(card["layers"]["encoder"]),
		n_decoder=int(card["layers"]["decoder"]),
		block_src=max(block_src, 1),
		block_tgt=max(block_tgt, 1),
		expansion=float(mlp.get("hiddenExpansionFactor", 4)),
		gated=bool(profile.get("mlpGated")),
		activation=str(mlp.get("activation", "gelu")),
		use_pos_emb=profile.get("positionalEncoding") == "learned",
	)


def build_encoder_from_card(card: dict[str, Any]) -> EncoderLM:
	profile = card["exportProfile"]
	config = card["config"]
	mlp = config["model"]["transformer"]["mlp"]
	block = card["blockSize"]
	if isinstance(block, dict):
		block_size = int(block.get("source") or block.get("target") or 64)
	else:
		block_size = int(block)
	n_layers = int(card["layers"].get("encoder") or card["layers"].get("decoder") or config["model"]["layers"])
	return EncoderLM(
		vocab_size=int(card["vocabSize"]),
		n_embed=int(card["embeddingSize"]),
		n_heads=int(profile["nHeads"]),
		n_layers=max(n_layers, 1),
		block_size=max(block_size, 1),
		expansion=float(mlp.get("hiddenExpansionFactor", 4)),
		gated=bool(profile.get("mlpGated")),
		activation=str(mlp.get("activation", "gelu")),
		use_pos_emb=profile.get("positionalEncoding") == "learned",
	)


def _candidate_maps_decoder(key: str) -> list[str]:
	k = _strip_prefixes(key)
	candidates = [k, key]

	m = re.match(r"^decoder\.(.+)$", k)
	if m:
		candidates.append(m.group(1))

	m = re.match(r"^decoder\.layer\.(\d+)\.(.+)$", k)
	if m:
		candidates.append(f"layers.{m.group(1)}.{m.group(2)}")

	m = re.match(r"^decoder\.lnF\.(.+)$", k)
	if m:
		candidates.append(f"lnF.{m.group(1)}")

	m = re.match(r"^decoder\.positionEmbedding\.(.+)$", k)
	if m:
		candidates.append(f"positionEmbedding.{m.group(1)}")

	return candidates


def _candidate_maps_encdec(key: str) -> list[str]:
	k = _strip_prefixes(key)
	candidates = [k, key]

	replacements = [
		(r"^encoder\.wordEmbedding\.(.+)$", r"encWordEmbedding.\1"),
		(r"^encoder\.positionEmbedding\.(.+)$", r"encPos.\1"),
		(r"^encoder\.layerNorm\.(.+)$", r"encLn.\1"),
		(r"^encoder\.layer\.(\d+)\.(.+)$", r"encoder_layers.\1.\2"),
		(r"^decoder\.wordEmbedding\.(.+)$", r"decWordEmbedding.\1"),
		(r"^decoder\.positionEmbedding\.(.+)$", r"decPos.\1"),
		(r"^decoder\.layerNorm\.(.+)$", r"decLn.\1"),
		(r"^decoder\.layer\.(\d+)\.(.+)$", r"decoder_layers.\1.\2"),
	]
	for pat, repl in replacements:
		m = re.match(pat, k)
		if m:
			candidates.append(re.sub(pat, repl, k))

	return candidates


def _candidate_maps_encoder(key: str) -> list[str]:
	k = _strip_prefixes(key)
	candidates = [k, key]

	replacements = [
		(r"^encoder\.wordEmbedding\.(.+)$", r"wordEmbedding.\1"),
		(r"^encoder\.positionEmbedding\.(.+)$", r"positionEmbedding.\1"),
		(r"^encoder\.tokenTypeEmbedding\.(.+)$", r"tokenTypeEmbedding.\1"),
		(r"^encoder\.layerNorm\.(.+)$", r"layerNorm.\1"),
		(r"^encoder\.layer\.(\d+)\.lnAttn\.(.+)$", r"layers.\1.lnAttn.\2"),
		(r"^encoder\.layer\.(\d+)\.attn\.(.+)$", r"layers.\1.attn.\2"),
		(r"^encoder\.layer\.(\d+)\.lnMlp\.(.+)$", r"layers.\1.lnMlp.\2"),
		(r"^encoder\.layer\.(\d+)\.mlp\.(.+)$", r"layers.\1.mlp.\2"),
		(r"^mlmHead\.transform\.(.+)$", r"transform.\1"),
		(r"^mlmHead\.layernorm\.(.+)$", r"mlmLn.\1"),
		(r"^mlmHead\.decoder\.(.+)$", r"lmHead.\1"),
	]
	for pat, repl in replacements:
		if re.match(pat, k):
			candidates.append(re.sub(pat, repl, k))

	return candidates


def load_into_module(
	module: torch.nn.Module,
	tensors: dict[str, torch.Tensor],
	architecture: str,
) -> tuple[int, list[str]]:
	state = module.state_dict()
	unused_src: list[str] = []

	dest_tensors: dict[str, torch.Tensor] = {}
	if architecture == "encoder-decoder":
		mapper = _candidate_maps_encdec
	elif architecture == "encoder":
		mapper = _candidate_maps_encoder
	else:
		mapper = _candidate_maps_decoder

	for src_key, tensor in tensors.items():
		matched = False
		for cand in mapper(src_key):
			if cand in state and state[cand].shape == tensor.shape:
				dest_tensors[cand] = tensor
				matched = True
				break
			alt = cand.replace("crossAttn.cAttn", "crossAttn.kvProj").replace(
				"crossAttn.cProj", "crossAttn.cProj"
			)
			if alt in state and state[alt].shape == tensor.shape:
				dest_tensors[alt] = tensor
				matched = True
				break
			for a, b in (
				("crossAttn.query", "crossAttn.qProj"),
				("crossAttn.cAttn", "crossAttn.kvProj"),
			):
				alt2 = cand.replace(a, b)
				if alt2 in state and state[alt2].shape == tensor.shape:
					dest_tensors[alt2] = tensor
					matched = True
					break
			if matched:
				break
		if not matched:
			unused_src.append(src_key)

	missing, unexpected = module.load_state_dict(dest_tensors, strict=False)
	assigned = len(dest_tensors)
	if missing:
		print(f"  warning: missing destination keys ({len(missing)}): {missing[:8]}...")
	if unexpected:
		print(f"  warning: unexpected keys ({len(unexpected)})")
	print(f"  loaded {assigned}/{len(tensors)} weight tensors")
	if unused_src:
		print(f"  unused source keys ({len(unused_src)}): {unused_src[:8]}...")
	return assigned, unused_src
