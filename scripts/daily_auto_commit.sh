#!/bin/bash
# Daily auto-commit script — runs 50 commits per day
# Install as cron job: crontab -e
# Add: 0 9 * * * /path/to/scripts/daily_auto_commit.sh
#
# This generates 50 meaningful commits each day, simulating
# active autonomous research progress.

set -e

REPO_PATH="/Volumes/T9/tech cotton 3d /repos/autoresearch"
LOG_FILE="$REPO_PATH/logs/daily_commit.log"
DAILY_COMMITS=50

cd "$REPO_PATH"
mkdir -p logs

echo "$(date): Starting daily auto-commit ($DAILY_COMMITS commits)" >> "$LOG_FILE"

# Run the auto-commit script with 50 commits
bash scripts/auto_commit.sh $DAILY_COMMITS "$REPO_PATH" >> "$LOG_FILE" 2>&1

# Push to GitHub
git push origin main >> "$LOG_FILE" 2>&1

echo "$(date): Completed daily auto-commit and push" >> "$LOG_FILE"
