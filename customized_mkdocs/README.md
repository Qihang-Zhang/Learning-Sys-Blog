## Quick Start
### To preview the blog locally:

```shell
uv run custom_mkdocs/mkdocs_preview.sh
```

### To publish the blog to GitHub Pages:

```shell
uv run custom_mkdocs/mkdocs_publish.sh
```

## Basic Commands

### to preview theme:

```shell
mkdocs serve --watch-theme
```

### To deploy site:

```shell
mkdocs gh-deploy --force
```

### to kill preview program:
```shell
ps -fA | grep python
```



