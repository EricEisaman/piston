#!/usr/bin/env bash
# Full-toolchain build for Browser Train (piston-train-toy).
# No git required. Skips wasm-opt / Binaryen.
#
# WASM skip:
#   - Auto-skips cargo + wasm-bindgen when target/pkg/piston-web matches a
#     fingerprint of Rust inputs (useful with Netlify/Render build caches).
#   - SKIP_WASM=1  force skip (requires existing pkg output)
#   - FORCE_WASM=1 force rebuild even when fingerprint matches
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Keep Cargo artifacts inside the repo so CI cache plugins can restore them.
export CARGO_HOME="${CARGO_HOME:-$ROOT/.cargo}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR"

CRATE="piston-web"
PROFILE="release"
OUT="target/pkg/${CRATE}"
FINGERPRINT_FILE="$OUT/.build-fingerprint"
# Must match [patch.crates-io] wasm-bindgen in Cargo.toml (EricEisaman fork).
WASM_BINDGEN_GIT="${WASM_BINDGEN_GIT:-https://github.com/EricEisaman/wasm-bindgen}"
WASM_BINDGEN_REV="${WASM_BINDGEN_REV:-4b4f9cd9731cf35725727bcac92940d51a559a50}"

hash_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@"
  else
    shasum -a 256 "$@"
  fi
}

compute_wasm_fingerprint() {
  # Stable digest of inputs that affect the piston-web WASM artifact.
  # Intentionally omits rustc --version so CI can match fingerprints before
  # rustup is on PATH (skip path never installs the toolchain).
  {
    echo "profile=${PROFILE}"
    echo "wasm_bindgen_rev=${WASM_BINDGEN_REV}"
    {
      [[ -f Cargo.toml ]] && printf '%s\n' Cargo.toml
      [[ -f Cargo.lock ]] && printf '%s\n' Cargo.lock
      find crates -type f \( -name Cargo.toml -o -name Cargo.lock -o -name '*.rs' \) 2>/dev/null
    } | LC_ALL=C sort -u | while IFS= read -r f; do
      [[ -f "$f" ]] || continue
      hash_cmd "$f"
    done
  } | hash_cmd | awk '{print $1}'
}

wasm_outputs_ok() {
  [[ -f "$OUT/${CRATE}_bg.wasm" && -f "$OUT/${CRATE}.js" && -f "$OUT/package.json" ]]
}

should_skip_wasm() {
  if [[ "${FORCE_WASM:-}" == "1" ]]; then
    echo "==> FORCE_WASM=1 — rebuilding WASM"
    return 1
  fi
  if [[ "${SKIP_WASM:-}" == "1" ]]; then
    if wasm_outputs_ok; then
      echo "==> SKIP_WASM=1 — reusing existing $OUT"
      return 0
    fi
    echo "ERROR: SKIP_WASM=1 but missing $OUT/${CRATE}_bg.wasm" >&2
    exit 1
  fi
  if ! wasm_outputs_ok; then
    return 1
  fi
  if [[ ! -f "$FINGERPRINT_FILE" ]]; then
    echo "==> No WASM fingerprint on disk — building"
    return 1
  fi
  local expected actual
  expected="$(cat "$FINGERPRINT_FILE")"
  actual="$(compute_wasm_fingerprint)"
  if [[ "$expected" == "$actual" ]]; then
    echo "==> WASM fingerprint match ($actual) — skipping cargo / wasm-bindgen"
    return 0
  fi
  echo "==> WASM fingerprint changed ($expected -> $actual) — rebuilding"
  return 1
}

echo "==> Preparing natural-language datasets"
bash "$ROOT/scripts/prepare-natural-data.sh"

if should_skip_wasm; then
  :
else
  echo "==> Ensuring Rust toolchain (nightly + wasm32-unknown-unknown)"
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  # Prefer project CARGO_HOME on PATH after install.
  export PATH="$CARGO_HOME/bin:$HOME/.cargo/bin:$PATH"
  rustup show >/dev/null
  rustup target add wasm32-unknown-unknown

  echo "==> Ensuring wasm-bindgen-cli matches ${WASM_BINDGEN_REV}"
  # Truncated crates.io trees (common when CI restores a bad .cargo/registry cache)
  # show up as rustc E0583 missing modules inside registry/src/.../memchr-*/src.
  purge_corrupt_cargo_registry_src() {
    local reg_src="${CARGO_HOME}/registry/src"
    [[ -d "$reg_src" ]] || return 0
    local sample
    sample="$(find "$reg_src" -path '*/memchr-*/src/lib.rs' 2>/dev/null | head -1 || true)"
    if [[ -n "$sample" ]]; then
      local memchr_src
      memchr_src="$(dirname "$sample")"
      if [[ ! -d "$memchr_src/arch" ]] || { [[ ! -f "$memchr_src/memmem.rs" ]] && [[ ! -d "$memchr_src/memmem" ]]; }; then
        echo "==> Corrupt Cargo registry src detected (incomplete memchr); wiping $reg_src"
        rm -rf "$reg_src"
      fi
    fi
  }
  install_wasm_bindgen_cli() {
    purge_corrupt_cargo_registry_src
    cargo install -f wasm-bindgen-cli \
      --git "$WASM_BINDGEN_GIT" \
      --rev "$WASM_BINDGEN_REV"
  }
  if command -v wasm-bindgen >/dev/null 2>&1 && [[ "${FORCE_WASM_BINDGEN:-}" != "1" ]]; then
    echo "==> wasm-bindgen already installed: $(wasm-bindgen --version) (set FORCE_WASM_BINDGEN=1 to reinstall)"
  else
    if ! install_wasm_bindgen_cli; then
      echo "==> wasm-bindgen-cli install failed; wiping registry src and retrying once" >&2
      rm -rf "${CARGO_HOME}/registry/src"
      install_wasm_bindgen_cli
    fi
  fi
  echo "==> wasm-bindgen: $(wasm-bindgen --version)"

  echo "==> Building ${CRATE} for wasm32 (${PROFILE})"
  cargo build \
    --manifest-path "./crates/${CRATE}/Cargo.toml" \
    --target wasm32-unknown-unknown \
    --"${PROFILE}"

  mkdir -p "$OUT"
  CRATE_UNDERSCORE="$(echo "$CRATE" | tr '-' '_')"
  wasm-bindgen "${CARGO_TARGET_DIR}/wasm32-unknown-unknown/${PROFILE}/${CRATE_UNDERSCORE}.wasm" \
    --target web \
    --out-dir "$OUT" \
    --out-name "$CRATE" \
    --reference-types

  cp "./crates/${CRATE}/package.json" "$OUT/package.json"

  if ! wasm_outputs_ok; then
    echo "ERROR: missing $OUT/${CRATE}_bg.wasm" >&2
    exit 1
  fi

  compute_wasm_fingerprint >"$FINGERPRINT_FILE"
  echo "==> Wrote WASM fingerprint $(cat "$FINGERPRINT_FILE")"
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

echo "==> Packing Browser Train ONNX toolkit zip (static download)"
bash "$ROOT/scripts/pack-export-inference-toolkit.sh"

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
