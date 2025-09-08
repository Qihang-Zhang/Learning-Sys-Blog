#!/usr/bin/env bash
set -euo pipefail

REPO_URL="git@github.com:Qihang-Zhang/customized_mkdocs.git"
TARGET_DIR="./customized_mkdocs"

# Remove existing directory if present
if [ -d "$TARGET_DIR" ]; then
  rm -rf "$TARGET_DIR"
fi

# Fresh clone
git clone "$REPO_URL" "$TARGET_DIR"

# Source aliases and copy config
chmod +x "$TARGET_DIR"
source "$TARGET_DIR/mkdocs_alias.sh"

$TARGET_DIR/maintain_config/copy_config.sh