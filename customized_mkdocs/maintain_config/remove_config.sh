#!/usr/bin/env bash

source ./customized_mkdocs/maintain_config/config_list.sh
for config in "${config_list[@]}"; do
    if [ -e "${config}" ]; then
        rm -rf "${config}"
    else
        echo "skip: ${config} not found" >&2
    fi
done