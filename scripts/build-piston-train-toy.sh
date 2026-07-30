#!/usr/bin/env bash
# Full-toolchain build for Browser Train (piston-train-toy).
# No git required. Skips wasm-opt / Binaryen.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Preparing natural-language datasets"
bash "$ROOT/scripts/prepare-natural-data.sh"


echo "==> Ensuring Rust toolchain (nightly + wasm32-unknown-unknown)"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
rustup show >/dev/null
rustup target add wasm32-unknown-unknown

# Must match [patch.crates-io] wasm-bindgen in Cargo.toml (EricEisaman fork).
WASM_BINDGEN_GIT="${WASM_BINDGEN_GIT:-https://github.com/EricEisaman/wasm-bindgen}"
WASM_BINDGEN_REV="${WASM_BINDGEN_REV:-4b4f9cd9731cf35725727bcac92940d51a559a50}"
echo "==> Ensuring wasm-bindgen-cli matches ${WASM_BINDGEN_REV}"
cargo install -f wasm-bindgen-cli \
  --git "$WASM_BINDGEN_GIT" \
  --rev "$WASM_BINDGEN_REV"
echo "==> wasm-bindgen: $(wasm-bindgen --version)"

CRATE="piston-web"
PROFILE="release"
OUT="target/pkg/${CRATE}"

echo "==> Building ${CRATE} for wasm32 (${PROFILE})"
cargo build \
  --manifest-path "./crates/${CRATE}/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --"${PROFILE}"

mkdir -p "$OUT"
CRATE_UNDERSCORE="$(echo "$CRATE" | tr '-' '_')"
wasm-bindgen "target/wasm32-unknown-unknown/${PROFILE}/${CRATE_UNDERSCORE}.wasm" \
  --target web \
  --out-dir "$OUT" \
  --out-name "$CRATE" \
  --reference-types

cp "./crates/${CRATE}/package.json" "$OUT/package.json"

if [[ ! -f "$OUT/${CRATE}_bg.wasm" ]]; then
  echo "ERROR: missing $OUT/${CRATE}_bg.wasm" >&2
  exit 1
fi

echo "==> Enabling Corepack / pnpm"
if command -v corepack >/dev/null 2>&1; then
  corepack enable
  corepack prepare pnpm@10.15.0 --activate
fi

echo "==> pnpm install"
pnpm install

echo "==> Building example-common"
pnpm --filter example-common build

echo "==> Building @piston-ml/piston-web"
pnpm --filter @piston-ml/piston-web build

echo "==> Building piston-train-toy (Browser Train)"
pnpm --filter piston-train-toy build

BUILD_DIR="examples/piston-train-toy/build"
if [[ ! -d "$BUILD_DIR" ]]; then
  echo "ERROR: missing $BUILD_DIR" >&2
  exit 1
fi
if [[ ! -d "examples/piston-train-toy/static/tokenizer" && ! -d "$BUILD_DIR/tokenizer" ]]; then
  echo "WARN: tokenizer static tree not found (toy datasets may still work)" >&2
fi

echo "==> Build complete: $BUILD_DIR"
