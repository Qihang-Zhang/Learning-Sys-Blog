import argparse
import sys
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc


def _infer_item_key(item: Any) -> Tuple[str, Any] | None:
    """Infer a stable, hashable identity for a list item.

    Heuristics:
      - If item is a mapping and has one of common id fields (name, scheme, id, key, type),
        use that field and its scalar value.
      - If item is a mapping with a single top-level key (e.g., mkdocs plugins: {blog: {...}}),
        use that key name as identity with a fixed tag ('single_key', key).
      - Otherwise, return None (no identity), so we fall back to override behavior.
    """
    if isinstance(item, dict):
        # Prefer explicit identifiers when scalar
        for field in ("name", "scheme", "id", "key", "type"):
            if field in item and isinstance(item[field], (str, int, float, bool)):
                return (field, item[field])
        # Single-key mapping: identify by the only key name (e.g., "blog")
        if len(item) == 1:
            only_key = next(iter(item))
            # Use a tagged identity to avoid colliding with explicit fields
            return ("single_key", str(only_key))
    return None


def _merge_lists(base: List[Any], customized: List[Any]) -> List[Any]:
    """Merge two lists with override semantics.

    Strategy:
      - If items have identifiable keys (via _infer_item_key), merge by identity:
        * If identity appears in both lists, deep-merge those items, customized wins.
        * If identity appears only in base, keep base item.
        * If identity appears only in customized, append customized item.
      - If items are not identifiable (scalars or mixed), fall back to `customized` overriding `base` entirely.
    """
    base_id_to_item: Dict[Tuple[str, Any], Any] = {}
    customized_id_to_item: Dict[Tuple[str, Any], Any] = {}

    base_all_identified = True
    for it in base:
        ident = _infer_item_key(it)
        if ident is None:
            base_all_identified = False
            break
        base_id_to_item[ident] = it

    customized_all_identified = True
    for it in customized:
        ident = _infer_item_key(it)
        if ident is None:
            customized_all_identified = False
            break
        customized_id_to_item[ident] = it

    # If either side has unidentifiable items, safest override behavior
    if not base_all_identified or not customized_all_identified:
        return customized

    merged: List[Any] = []

    # Keep order preference: start with base order but override conflicts with customized
    for ident, base_item in base_id_to_item.items():
        if ident in customized_id_to_item:
            merged.append(deep_merge(base_item, customized_id_to_item[ident]))
        else:
            merged.append(base_item)

    # Append any new items from customized not present in base
    for ident, cust_item in customized_id_to_item.items():
        if ident not in base_id_to_item:
            merged.append(cust_item)

    return merged


def _normalize_mkdocs_structure(data: Any, *, is_root: bool) -> Any:
    """Only at root-level, move top-level theme-related keys under 'theme'."""
    if not isinstance(data, dict):
        return data

    if is_root:
        theme_keys = {"palette", "features", "language", "icon", "name", "logo", "favicon"}
        top_level_theme_items = {k: v for k, v in list(data.items()) if k in theme_keys}
        if top_level_theme_items:
            theme_obj = data.get("theme")
            if not isinstance(theme_obj, dict):
                theme_obj = {} if theme_obj is None else {"value": theme_obj}
            for k, v in top_level_theme_items.items():
                theme_obj[k] = v
                del data[k]
            data["theme"] = theme_obj

    # Recurse into nested dicts and lists
    for k, v in list(data.items()):
        if isinstance(v, dict):
            data[k] = _normalize_mkdocs_structure(v, is_root=False)
        elif isinstance(v, list):
            data[k] = [
                _normalize_mkdocs_structure(item, is_root=False) if isinstance(item, dict) else item
                for item in v
            ]

    return data


def deep_merge(base: Any, customized: Any) -> Any:
    """Deep-merge two YAML-loaded Python objects.

    - If both are dicts: recursively merge; on conflict, `customized` wins.
    - If both are lists: try key-aware merge; otherwise, `customized` replaces `base`.
    - Otherwise: return `customized`.
    """
    if isinstance(base, dict) and isinstance(customized, dict):
        merged: Dict[str, Any] = {}
        # keys from both; customized overrides
        for key in base.keys() | customized.keys():
            if key in base and key in customized:
                merged[key] = deep_merge(base[key], customized[key])
            elif key in customized:
                merged[key] = customized[key]
            else:
                merged[key] = base[key]
        return merged

    if isinstance(base, list) and isinstance(customized, list):
        return _merge_lists(base, customized)

    return customized


def load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def dump_yaml(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def parse_args(argv: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two YAML files where customized overrides base on conflicts."
        )
    )
    parser.add_argument(
        "-b",
        "--base",
        required=True,
        help="Path to base YAML file",
    )
    parser.add_argument(
        "-c",
        "--customized",
        required=True,
        help="Path to customized YAML file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to write merged YAML output",
    )
    return parser.parse_args(argv)


def main(argv: Any = None) -> int:
    args = parse_args(argv)

    base_data = load_yaml(args.base)
    customized_data = load_yaml(args.customized)

    base_data = _normalize_mkdocs_structure(base_data, is_root=True)
    customized_data = _normalize_mkdocs_structure(customized_data, is_root=True)

    if base_data is None:
        merged = customized_data
    elif customized_data is None:
        merged = base_data
    else:
        merged = deep_merge(base_data, customized_data)

    dump_yaml(merged, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
