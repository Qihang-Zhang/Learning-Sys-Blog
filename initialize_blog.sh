#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Qihang-Zhang/fused_mkdocs.git"
TARGET_DIR="./fused_mkdocs"

# Remove existing directory if present
if [ -d "$TARGET_DIR" ]; then
  rm -rf "$TARGET_DIR"
fi

# Fresh clone
git clone "$REPO_URL" "$TARGET_DIR"

zsh ./fused_mkdocs/maintain_config/copy_config.sh

