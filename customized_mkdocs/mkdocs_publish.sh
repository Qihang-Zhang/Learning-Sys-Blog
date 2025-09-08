#!/bin/bash

# default commit message
commit_msg="auto update"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--message)
            commit_msg="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

source .venv/bin/activate
./customized_mkdocs/mkdocs_genyml.sh -p
mkdocs gh-deploy --force
git add .
git commit -m "$commit_msg"
git push