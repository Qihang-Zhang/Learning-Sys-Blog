#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Qihang-Zhang/customized_mkdocs.git"
TARGET_DIR="./customized_mkdocs"

# Remove existing directory if present
if [ -d "$TARGET_DIR" ]; then
  rm -rf "$TARGET_DIR"
fi

# Fresh clone
git clone "$REPO_URL" "$TARGET_DIR"

# Source aliases and copy config
source "$TARGET_DIR/mkdocs_alias.sh"
copy_mkdocs_config