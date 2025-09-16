#!/usr/bin/env bash
set -euo pipefail

REPO_URL="git@github.com:Qihang-Zhang/Comping.git"
TARGET_DIR="./customized_mkdocs"

# Remove existing directory if present
if [ -d "$TARGET_DIR" ]; then
  rm -rf "$TARGET_DIR"
fi

# Fresh clone
git clone "$REPO_URL" "$TARGET_DIR"

zsh ./customized_mkdocs/maintain_config/copy_config.sh

