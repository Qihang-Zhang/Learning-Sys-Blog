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

python3 ./customized_mkdocs/yml_add_ifpublish.py \
    --will-publish ${publish} \
    -i mkdocs_config/mkdocs_config.yml \
    -o tmp/mkdocs_config.yml

python3 ./customized_mkdocs/yml_merge.py \
    -b customized_mkdocs/base_mkdocs.yml \
    -c tmp/mkdocs_config.yml \
    -o tmp/mkdocs.yml

python3 ./customized_mkdocs/yml_add_nav.py \
    -i tmp/mkdocs.yml \
    -o mkdocs.yml

# rm -rf tmp