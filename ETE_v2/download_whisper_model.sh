#!/bin/bash
# Download the faster-whisper CT2 model (same as FSMv3.1_R2)
# This is the whisper-large-v3-turbo model converted to CTranslate2 format

set -e

MODEL_DIR="models/whisper-large-v3-turbo-ct2"

if [ -d "$MODEL_DIR" ]; then
    echo "Model already exists at $MODEL_DIR"
    exit 0
fi

echo "Installing dependencies..."
pip install -q faster-whisper huggingface_hub ctranslate2

echo "Downloading whisper-large-v3-turbo CT2 model..."
mkdir -p models

# Use huggingface_hub to download the CT2-converted model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Systran/faster-whisper-large-v3',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
)
print('Model downloaded successfully to $MODEL_DIR')
"

echo ""
echo "Done! Model is at: $MODEL_DIR"
echo "Restart the server and select 'Mic: Whisper' to use it."
