#!/usr/bin/env bash
# Pulls vasista22/whisper-hindi-small from Hugging Face and converts to CT2 INT8.
# Keeps the existing whisper-large-v3-turbo-ct2 directory in place as Tier-2 verifier.
set -euo pipefail

# Use the project venv binaries (python, ct2-transformers-converter) without requiring activation.
export PATH="/Users/shantanuchandra/Downloads/eye_test_engine_Claude/ETE_v2/.venv/bin:$PATH"

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"
mkdir -p "$MODELS_DIR"

VASISTA_HF_DIR="$MODELS_DIR/vasista22-whisper-hindi-small-hf"
VASISTA_CT2_DIR="$MODELS_DIR/vasista22-whisper-hindi-small-ct2"

if [ ! -d "$VASISTA_HF_DIR" ]; then
  echo "[1/2] Downloading vasista22/whisper-hindi-small from HF..."
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='vasista22/whisper-hindi-small', local_dir='$VASISTA_HF_DIR', local_dir_use_symlinks=False)
"
else
  echo "[1/2] HF checkpoint already present at $VASISTA_HF_DIR"
fi

# vasista22/whisper-hindi-small ships only the slow-tokenizer artifacts (vocab.json,
# merges.txt). faster-whisper / ct2-transformers-converter need the fast tokenizer.json,
# so materialize it from the slow tokenizer if missing.
if [ ! -f "$VASISTA_HF_DIR/tokenizer.json" ]; then
  echo "[1.5/2] Generating tokenizer.json from slow tokenizer..."
  python -c "
from transformers import WhisperTokenizerFast
tok = WhisperTokenizerFast.from_pretrained('$VASISTA_HF_DIR')
tok.save_pretrained('$VASISTA_HF_DIR')
"
fi

if [ ! -d "$VASISTA_CT2_DIR" ]; then
  echo "[2/2] Converting to CT2 INT8 at $VASISTA_CT2_DIR..."
  ct2-transformers-converter \
    --model "$VASISTA_HF_DIR" \
    --output_dir "$VASISTA_CT2_DIR" \
    --quantization int8 \
    --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json
else
  echo "[2/2] CT2 directory already present at $VASISTA_CT2_DIR"
fi

echo "Done. CT2 model: $VASISTA_CT2_DIR"
