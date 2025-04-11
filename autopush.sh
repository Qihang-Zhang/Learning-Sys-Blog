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
python ./utils/yml_autobuild.py --mode published
mkdocs gh-deploy --force
git add .
git commit -m "$commit_msg"
git push