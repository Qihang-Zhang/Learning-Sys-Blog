source .venv/bin/activate

publish=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--publish)
      publish=true
      shift
      ;;
    *)
      echo "Usage: $0 [-p|--publish]"
      exit 1
      ;;
  esac
done

mkdir -p tmp

python ./customized_mkdocs/add_ifpublish_to_yml.py \
    --will-publish ${publish} \
    -i mkdocs_config/mkdocs_config.yml \
    -o tmp/mkdocs_config.yml

python ./customized_mkdocs/merge_ymls.py \
    -b customized_mkdocs/base_mkdocs.yml \
    -c tmp/mkdocs_config.yml \
    -o tmp/mkdocs.yml

python ./customized_mkdocs/add_nav_to_yml.py \
    -i tmp/mkdocs.yml \
    -o mkdocs.yml

# rm -rf tmp