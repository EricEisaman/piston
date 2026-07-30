#!/usr/bin/env bash
# Pack a self-contained Browser Train → ONNX toolkit zip for site users.
# No repo clone required. Does NOT include a .venv (users run setup locally).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/scripts/export_inference"
OUT_ZIP="$ROOT/examples/piston-train-toy/static/browser-train-onnx-toolkit.zip"
NAME="browser-train-onnx-toolkit"
STAGING="$(mktemp -d "${TMPDIR:-/tmp}/${NAME}.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

DEST="$STAGING/$NAME"
mkdir -p "$DEST/export_inference"

echo "==> Packing ONNX toolkit from $SRC"

# Python package (omit developer README; ship end-user README below)
cp "$SRC/__init__.py" "$DEST/export_inference/"
cp "$SRC/__main__.py" "$DEST/export_inference/"
cp "$SRC/convert.py" "$DEST/export_inference/"
cp "$SRC/models.py" "$DEST/export_inference/"
cp "$SRC/validate.py" "$DEST/export_inference/"
cp "$SRC/weights.py" "$DEST/export_inference/"
cp "$SRC/transformers_js.py" "$DEST/export_inference/"
cp "$SRC/requirements.txt" "$DEST/requirements.txt"

cat >"$DEST/README.md" <<'EOF'
# Browser Train → ONNX / Transformers.js toolkit

Convert a purple **ONNX** download from Browser Train into packages for
onnxruntime-web **and** Transformers.js. You do **not** need the full piston repository.

## Requirements

- Python 3.10+ on your machine (`python3 --version`)
- The two files from Browser Train’s purple ONNX download:
  - `{run}.inference.safetensors`
  - `{run}.model.json`

## One-time setup

Creates a local `.venv` in this folder and installs dependencies (includes PyTorch; large, once).

**macOS / Linux**

```bash
chmod +x setup.sh convert.sh
./setup.sh
```

**Windows (Command Prompt)**

```bat
setup.bat
```

## Convert (default: both targets)

**macOS / Linux**

```bash
./convert.sh ~/Downloads/my-run.inference.safetensors -o ./out
```

**Windows**

```bat
convert.bat %USERPROFILE%\Downloads\my-run.inference.safetensors -o out
```

If `my-run.model.json` sits next to the `.inference.safetensors` file, it is picked up
automatically. Otherwise pass `--model-json path\to\my-run.model.json`.

Default output (both targets):

| Path | Use with |
|------|----------|
| `out/ort/` | onnxruntime-web (`browser-train-infer`: `loadModel` / `complete` / `encodeDecode`) |
| `out/transformers-js/` | `@huggingface/transformers` (`browser-train-infer-tjs`, decoder-only) |

Optional: `--targets ort` or `--targets transformers-js` for a single layout.

## Next steps

- **ORT:** copy `out/ort/*` into a small onnxruntime-web app’s `public/model/`
  (see `examples/browser-train-infer`).
- **Transformers.js (decoder):** copy `out/transformers-js/*` into
  `public/models/browser-train/` (see `examples/browser-train-infer-tjs`).
- **Encoder-decoder toys:** generation stays on the ORT path; the Transformers.js
  folder is packaging/tokenizer only (no Seq2Seq AutoModel in v1).

Full visual guide + per-preset copy-paste snippets: Browser Train → **Docs** tab.

## Notes

- Do not commit or share the `.venv` folder; re-run `setup` on each machine.
- Encoder-only (Dyck / MLM) and RNN export are not supported in this toolkit yet.
EOF

cat >"$DEST/setup.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ and try again." >&2
  exit 1
fi

echo "==> Creating .venv in $ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "==> Setup complete."
echo "Convert with:"
echo "  ./convert.sh ~/Downloads/your-run.inference.safetensors -o ./out"
EOF

cat >"$DEST/convert.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: .venv not found. Run ./setup.sh first." >&2
  exit 1
fi

if [[ "$#" -lt 1 ]]; then
  echo "Usage: ./convert.sh path/to/run.inference.safetensors -o ./out [--model-json path/to/run.model.json] [--targets both|ort|transformers-js]" >&2
  exit 1
fi

export PYTHONPATH="$ROOT"
exec "$PY" -m export_inference convert "$@"
EOF

cat >"$DEST/setup.bat" <<'EOF'
@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found. Install Python 3.10+ and try again.
  exit /b 1
)

echo ==^> Creating .venv in %CD%
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo ==^> Setup complete.
echo Convert with:
echo   convert.bat %%USERPROFILE%%\Downloads\your-run.inference.safetensors -o out
endlocal
EOF

cat >"$DEST/convert.bat" <<'EOF'
@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run setup.bat first.
  exit /b 1
)

if "%~1"=="" (
  echo Usage: convert.bat path\to\run.inference.safetensors -o out [--model-json path\to\run.model.json] [--targets both]
  exit /b 1
)

set PYTHONPATH=%CD%
".venv\Scripts\python.exe" -m export_inference convert %*
endlocal
EOF

chmod +x "$DEST/setup.sh" "$DEST/convert.sh"

mkdir -p "$(dirname "$OUT_ZIP")"
rm -f "$OUT_ZIP"

# Prefer zip(1); fall back to Python if unavailable (e.g. minimal CI images).
if command -v zip >/dev/null 2>&1; then
  (cd "$STAGING" && zip -r -q "$OUT_ZIP" "$NAME")
else
  TOOLKIT_STAGING="$STAGING" TOOLKIT_OUT_ZIP="$OUT_ZIP" TOOLKIT_NAME="$NAME" python3 - <<'PY'
import os
import pathlib
import zipfile

staging = pathlib.Path(os.environ["TOOLKIT_STAGING"])
out = pathlib.Path(os.environ["TOOLKIT_OUT_ZIP"])
name = os.environ["TOOLKIT_NAME"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    root = staging / name
    for path in root.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(staging).as_posix())
print(f"wrote {out}")
PY
fi

BYTES="$(wc -c <"$OUT_ZIP" | tr -d ' ')"
echo "==> Wrote $OUT_ZIP ($BYTES bytes)"
