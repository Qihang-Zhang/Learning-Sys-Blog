#!/usr/bin/env bash

source ./customized_mkdocs/maintain_config/config_list.sh
for config in "${config_list[@]}"; do
    if [ -e "customized_mkdocs/${config}" ]; then
        cp -r "customized_mkdocs/${config}" .
    else
        echo "skip: customized_mkdocs/${config} not found" >&2
    fi
done