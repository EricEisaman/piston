/**
 * Colorized Mermaid architecture diagrams for each visible Browser Train preset.
 * Uses Mermaid classDef fills/strokes (not plain unstyled flowcharts).
 */

export type ArchitectureDiagram = {
	presetId: string;
	summary: string;
	/** Mermaid source including classDef color styles */
	mermaid: string;
};

/** Shared palette applied to every diagram via classDef + :::class */
const MERMAID_CLASS_DEFS = `
classDef input fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:2px
classDef embed fill:#fce7f3,stroke:#db2777,color:#9d174d,stroke-width:2px
classDef attn fill:#ede9fe,stroke:#7c3aed,color:#5b21b6,stroke-width:2px
classDef mlp fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px
classDef enc fill:#e0f2fe,stroke:#0284c7,color:#075985,stroke-width:2px
classDef dec fill:#fef3c7,stroke:#d97706,color:#92400e,stroke-width:2px
classDef cross fill:#fae8ff,stroke:#c026d3,color:#86198f,stroke-width:2px
classDef out fill:#ffedd5,stroke:#ea580c,color:#9a3412,stroke-width:2px
classDef data fill:#f1f5f9,stroke:#64748b,color:#334155,stroke-width:2px
classDef tip fill:#ecfdf5,stroke:#059669,color:#065f46,stroke-width:2px
classDef warn fill:#fff7ed,stroke:#ea580c,color:#9a3412,stroke-width:2px
`.trim();

const withStyles = (body: string) => `${body.trim()}\n\n${MERMAID_CLASS_DEFS}`;

const ENC_DEC_TOY = withStyles(`
flowchart TB
  subgraph dataBlock ["Toy task"]
    Src["Source tokens"]:::input
    Tgt["Target tokens"]:::input
  end
  EmbE["Encoder embedding"]:::embed
  EmbD["Decoder embedding"]:::embed
  Enc["Encoder stack<br/>self-attention + MLP"]:::enc
  Dec["Decoder stack<br/>masked self-attn + MLP"]:::dec
  XAttn["Cross-attention"]:::cross
  Head["LM head to vocab"]:::out
  Src --> EmbE --> Enc
  Tgt --> EmbD --> Dec
  Enc --> XAttn
  Dec --> XAttn
  XAttn --> Head
`);

const ENC_DEC_ONNX = withStyles(`
flowchart TB
  subgraph dataBlock ["Toy task - ONNX-friendly"]
    Src["Source tokens"]:::input
    Tgt["Target tokens"]:::input
  end
  EmbE["Encoder embedding<br/>learned PE"]:::embed
  EmbD["Decoder embedding<br/>learned PE"]:::embed
  Enc["Encoder - standard GELU MLP<br/>no GQA / gating / qkNorm"]:::enc
  Dec["Decoder - standard GELU MLP"]:::dec
  XAttn["Cross-attention"]:::cross
  Head["LM head to vocab"]:::out
  Tip["Purple ONNX export OK"]:::tip
  Src --> EmbE --> Enc
  Tgt --> EmbD --> Dec
  Enc --> XAttn
  Dec --> XAttn
  XAttn --> Head --> Tip
`);

const DECODER_TINY = withStyles(`
flowchart LR
  Tok["Token stream<br/>TinyStories BPE"]:::data
  Emb["Token + pos embed"]:::embed
  Blk["Nx decoder blocks<br/>causal self-attn + MLP"]:::attn
  Mlp["Feed-forward"]:::mlp
  Head["LM head"]:::out
  Next["Next-token logits"]:::out
  Tok --> Emb --> Blk
  Blk --> Mlp
  Blk --> Head --> Next
`);

const DECODER_TINY_ONNX = withStyles(`
flowchart LR
  Tok["Token stream<br/>TinyStories BPE"]:::data
  Emb["Embed + learned PE"]:::embed
  Blk["6x causal blocks<br/>export-safe attn"]:::attn
  Mlp["Standard GELU MLP"]:::mlp
  Head["LM head"]:::out
  Tip["ORT + Transformers.js"]:::tip
  Tok --> Emb --> Blk --> Mlp
  Blk --> Head --> Tip
`);

const DECODER_FINEWEB = withStyles(`
flowchart TB
  Data["FineWeb shards<br/>BPE vocab 8192"]:::data
  Emb["Embed dim 768"]:::embed
  Blk["12x GPT-2-shaped<br/>decoder blocks"]:::attn
  Mlp["GELU MLP 4x expand"]:::mlp
  Head["LM head"]:::out
  Warn["Large VRAM - slow steps"]:::warn
  Data --> Emb --> Blk --> Mlp
  Blk --> Head
  Head --- Warn
`);

const DECODER_FINEWEB_ONNX = withStyles(`
flowchart TB
  Data["FineWeb - exportable"]:::data
  Emb["Embed + learned PE<br/>dim 768"]:::embed
  Blk["12x causal blocks<br/>no GQA / gating / qkNorm"]:::attn
  Mlp["Standard GELU MLP"]:::mlp
  Head["LM head"]:::out
  Tip["Purple ONNX to GPT-2 style"]:::tip
  Data --> Emb --> Blk --> Mlp
  Blk --> Head --> Tip
`);

const LIL_SIGGY = withStyles(`
flowchart TB
  Upload["Upload .txt / .md"]:::input
  EncTok["FineWeb BPE 8192 encode"]:::embed
  Shards["llm.c shards in IndexedDB"]:::data
  Emb["Embed dim 768"]:::embed
  Blk["12x decoder · GQA 12Q/6KV<br/>gating + qkNorm"]:::attn
  Mlp["GELU MLP"]:::mlp
  Head["LM head"]:::out
  Complete["Metrics completions"]:::tip
  Upload --> EncTok --> Shards --> Emb --> Blk --> Mlp
  Blk --> Head --> Complete
`);

const LIL_SIGGY_ONNX = withStyles(`
flowchart TB
  Upload["Same Lil Siggy corpus"]:::input
  Emb["Embed + learned PE"]:::embed
  Blk["12x causal blocks<br/>export-safe attn"]:::attn
  Mlp["Standard GELU MLP"]:::mlp
  Head["LM head"]:::out
  Tip["Purple ONNX + tokenizer.json"]:::tip
  Upload --> Emb --> Blk --> Mlp
  Blk --> Head --> Tip
`);

const ENCODER_DYCK = withStyles(`
flowchart LR
  Seq["Dyck sequence<br/>brackets / parens"]:::input
  Emb["Token embedding"]:::embed
  Enc["Encoder stack<br/>bidirectional self-attn"]:::enc
  Mlp["MLP per block"]:::mlp
  MLM["Masked LM head"]:::out
  Seq --> Emb --> Enc --> Mlp
  Enc --> MLM
`);

const ONNX_MODIFIER = withStyles(`
flowchart TB
  Base["Any transformer preset"]:::data
  Mod["onnx-export-friendly layer"]:::tip
  Off["Disable GQA / gating / sinks<br/>qkNorm / softcap"]:::warn
  On["Learned PE + GELU MLP"]:::mlp
  Out["Inference-exportable graph"]:::out
  Base --> Mod
  Mod --> Off
  Mod --> On
  Off --> Out
  On --> Out
`);

export const ARCHITECTURE_DIAGRAMS: ArchitectureDiagram[] = [
	{
		presetId: 'sort-characters',
		summary: 'Encoder–decoder transformer that sorts character sequences (toy seq2seq).',
		mermaid: ENC_DEC_TOY
	},
	{
		presetId: 'sort-characters-onnx',
		summary: 'Same sort-characters topology with ONNX-export-friendly attention/MLP settings.',
		mermaid: ENC_DEC_ONNX
	},
	{
		presetId: 'reverse-sequence',
		summary: 'Encoder–decoder that reverses the input sequence.',
		mermaid: ENC_DEC_TOY
	},
	{
		presetId: 'two-sum',
		summary: 'Encoder–decoder toy: find numbers that add to a target sum.',
		mermaid: ENC_DEC_TOY
	},
	{
		presetId: 'dyck-encoder',
		summary: 'Encoder-only (BERT-style MLM) for Dyck / balanced-parentheses languages.',
		mermaid: ENCODER_DYCK
	},
	{
		presetId: 'tinystories',
		summary: 'Decoder-only GPT-style LM on TinyStories (~15M parameters).',
		mermaid: DECODER_TINY
	},
	{
		presetId: 'tinystories-onnx',
		summary: 'TinyStories decoder with export-safe PE / MLP / attention flags.',
		mermaid: DECODER_TINY_ONNX
	},
	{
		presetId: 'fineweb',
		summary: 'GPT-2-sized decoder on FineWeb; heavy VRAM — prefer fineweb-onnx to export.',
		mermaid: DECODER_FINEWEB
	},
	{
		presetId: 'fineweb-onnx',
		summary: 'FineWeb GPT-2-shaped decoder configured for purple ONNX inference export.',
		mermaid: DECODER_FINEWEB_ONNX
	},
	{
		presetId: 'lil-siggy',
		summary:
			'Custom-corpus GPT-2-sized decoder with GQA (12Q/6KV), attention gating, and qkNorm. Upload text in Dataset controls.',
		mermaid: LIL_SIGGY
	},
	{
		presetId: 'lil-siggy-onnx',
		summary:
			'Lil Siggy export sibling: onnx-export-friendly overlay + tokenizer.json with purple download.',
		mermaid: LIL_SIGGY_ONNX
	},
	{
		presetId: 'onnx-export-friendly',
		summary: 'Modifier layer: snaps transformer settings to an ONNX-friendly subset.',
		mermaid: ONNX_MODIFIER
	}
];

export function getArchitectureDiagram(presetId: string): ArchitectureDiagram | undefined {
	return ARCHITECTURE_DIAGRAMS.find((d) => d.presetId === presetId);
}
