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

python ./utils/yml_autobuild.py
mkdocs gh-deploy --force
git add .
git commit -m "$commit_msg"
git push