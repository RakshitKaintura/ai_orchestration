#!/bin/bash
# scripts/push_to_github.sh
# Run this AFTER creating the GitHub repo at https://github.com/new

set -e

REPO_URL="$1"  # e.g., https://github.com/YOUR_USERNAME/mega-ai.git

if [ -z "$REPO_URL" ]; then
  echo "Usage: bash scripts/push_to_github.sh https://github.com/YOUR_USERNAME/mega-ai.git"
  exit 1
fi

echo "Adding remote: $REPO_URL"
git remote add origin "$REPO_URL"

echo "Pushing all commits..."
git push -u origin master

echo "Done! Repository is live at: ${REPO_URL%.git}"
