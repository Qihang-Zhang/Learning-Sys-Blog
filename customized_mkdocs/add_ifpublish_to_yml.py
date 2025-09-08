import argparse
import sys
from typing import Any

try:
    import yaml  # noqa: F401  # imported to ensure helpful error if PyYAML missing
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc


# Reuse consistent YAML IO/formatting from the merge utility if available
try:  # pragma: no cover - convenience import
    from customized_mkdocs.merge_yml import load_yaml, dump_yaml  # type: ignore
except Exception:  # pragma: no cover - fallback minimal IO
    def load_yaml(path: str) -> Any:
        import yaml as _yaml

        with open(path, "r", encoding="utf-8") as f:
            return _yaml.safe_load(f)

    def dump_yaml(data: Any, path: str) -> None:
        import yaml as _yaml

        dumped = _yaml.safe_dump(
            data,
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(dumped if dumped.endswith("\n") else dumped + "\n")


def _append_subfolder_to_path(original: str, subfolder: str) -> str:
    normalized = original.rstrip("/")
    suffix = subfolder.strip().strip("/")
    if not suffix:
        return normalized
    return f"{normalized}/{suffix}"


def _update_blog_post_dir(config: Any, subfolder: str) -> bool:
    """If config contains plugins -> blog -> post_dir, append subfolder.

    Returns True if an update was made.
    """
    if not isinstance(config, dict):
        return False

    plugins = config.get("plugins")
    if not isinstance(plugins, list):
        return False

    updated = False
    for idx, plugin in enumerate(plugins):
        if not isinstance(plugin, dict):
            continue
        if "blog" not in plugin:
            continue
        blog_cfg = plugin.get("blog")
        if not isinstance(blog_cfg, dict):
            continue
        if "post_dir" not in blog_cfg:
            continue
        post_dir_val = blog_cfg.get("post_dir")
        if isinstance(post_dir_val, str):
            blog_cfg["post_dir"] = _append_subfolder_to_path(post_dir_val, subfolder)
            # ensure the updated mapping is placed back (not strictly necessary for dicts)
            plugins[idx] = {**plugin, "blog": blog_cfg}
            updated = True
    return updated


def _str_to_bool(value: str) -> bool:
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}
    val = value.strip().lower()
    if val in truthy:
        return True
    if val in falsy:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value for --will-publish: '{value}'. Use true/false."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite an input mkdocs YAML to output, appending a subfolder to "
            "plugins -> blog -> post_dir if present."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input YAML file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to write output YAML"
    )
    parser.add_argument(
        "--will-publish",
        type=_str_to_bool,
        default=False,
        metavar="{true,false}",
        help="Whether to append the provided subfolder to blog post_dir",
    )
    parser.add_argument(
        "-s",
        "--subfolder",
        type=str,
        default="published",
        help=(
            "Subfolder name to append to blog post_dir (e.g., 'published' or 'drafts')"
        ),
    )
    args = parser.parse_args()

    data = load_yaml(args.input)
    # Only attempt update if structure matches and will-publish is true
    if args.will_publish:
        _update_blog_post_dir(data, args.subfolder)
    dump_yaml(data, args.output)


