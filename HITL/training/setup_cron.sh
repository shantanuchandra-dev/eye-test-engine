#!/bin/bash
# Setup cron jobs for weekly retraining and daily sync.
#
# Usage:
#   bash voice/training/setup_cron.sh          # preview cron entries
#   bash voice/training/setup_cron.sh --apply  # install to crontab

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: venv not found at $PROJECT_DIR/venv"
    echo "Create it with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

CRON_ENTRIES="
# Eye Test Engine — Weekly Retrain (Monday 3am)
0 3 * * 1 cd $PROJECT_DIR && $VENV_PYTHON -m voice.training.weekly_retrain >> $HOME/.eye_test_audio/_analysis/cron_retrain.log 2>&1

# Eye Test Engine — Daily Sync Push (2am)
# Uncomment after configuring SYNC_SERVER_URL or SYNC_RSYNC_TARGET
# 0 2 * * * cd $PROJECT_DIR && SYNC_SERVER_URL=https://central.example.com SYNC_API_KEY=xxx SYNC_CLINIC_ID=clinic_01 $VENV_PYTHON -m voice.training.server_sync --push >> $HOME/.eye_test_audio/_analysis/cron_sync.log 2>&1

# Eye Test Engine — Daily Sync Pull (4am)
# 0 4 * * * cd $PROJECT_DIR && SYNC_SERVER_URL=https://central.example.com SYNC_API_KEY=xxx SYNC_CLINIC_ID=clinic_01 $VENV_PYTHON -m voice.training.server_sync --pull >> $HOME/.eye_test_audio/_analysis/cron_sync.log 2>&1
"

echo "=== Cron entries for Eye Test Engine ==="
echo "$CRON_ENTRIES"

if [ "$1" = "--apply" ]; then
    # Add to existing crontab without overwriting
    (crontab -l 2>/dev/null; echo "$CRON_ENTRIES") | crontab -
    echo ""
    echo "✓ Cron jobs installed. Verify with: crontab -l"
    echo "  Weekly retrain log: ~/.eye_test_audio/_analysis/cron_retrain.log"
else
    echo ""
    echo "Preview only. Run with --apply to install."
fi
