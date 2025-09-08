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
    if isinstance(item, dict):
        for field in ("name", "scheme", "id", "key", "type"):
            if field in item and isinstance(item[field], (str, int, float, bool)):
                return (field, item[field])
        if len(item) == 1:
            only_key = next(iter(item))
            return ("single_key", str(only_key))
    return None


def _hash_token(item: Any) -> Tuple[str, Any]:
    """Create a stable token for order-preserving set semantics for non-identifiable items.

    Scalars use their type-tag and value; mappings/sequences are serialized.
    """
    if isinstance(item, (str, int, float, bool)) or item is None:
        return (type(item).__name__, item)
    # Fallback to YAML-stable serialization
    serialized = yaml.safe_dump(item, sort_keys=True, default_flow_style=True)
    return ("yaml", serialized)


def _merge_lists(base: List[Any], customized: List[Any]) -> List[Any]:
    # Split into identifiable items and others for both lists
    base_id_map: Dict[Tuple[str, Any], Any] = {}
    base_other_tokens: List[Tuple[str, Any]] = []
    base_token_to_item: Dict[Tuple[str, Any], Any] = {}

    for it in base:
        ident = _infer_item_key(it)
        if ident is not None:
            base_id_map[ident] = it
        else:
            tok = _hash_token(it)
            base_other_tokens.append(tok)
            base_token_to_item[tok] = it

    cust_id_map: Dict[Tuple[str, Any], Any] = {}
    cust_other_tokens: List[Tuple[str, Any]] = []
    cust_token_to_item: Dict[Tuple[str, Any], Any] = {}

    for it in customized:
        ident = _infer_item_key(it)
        if ident is not None:
            cust_id_map[ident] = it
        else:
            tok = _hash_token(it)
            cust_other_tokens.append(tok)
            cust_token_to_item[tok] = it

    # Build the merged list preserving base order where possible
    merged: List[Any] = []
    seen_idents: set[Tuple[str, Any]] = set()
    seen_tokens: set[Tuple[str, Any]] = set()

    # Iterate original base list order and place merged items accordingly
    for it in base:
        ident = _infer_item_key(it)
        if ident is not None:
            if ident in cust_id_map:
                merged.append(deep_merge(base_id_map[ident], cust_id_map[ident]))
            else:
                merged.append(base_id_map[ident])
            seen_idents.add(ident)
        else:
            tok = _hash_token(it)
            # If customized has an equal token, prefer customized instance (could differ by formatting)
            if tok in cust_token_to_item:
                merged.append(cust_token_to_item[tok])
            else:
                merged.append(base_token_to_item[tok])
            seen_tokens.add(tok)

    # Append any new customized-only identifiable items, preserving their customized order
    for it in customized:
        ident = _infer_item_key(it)
        if ident is not None and ident not in seen_idents and ident not in base_id_map:
            merged.append(cust_id_map[ident])
            seen_idents.add(ident)

    # Append any new customized-only other tokens, preserving order
    for tok in cust_other_tokens:
        if tok not in seen_tokens and tok not in base_token_to_item:
            merged.append(cust_token_to_item[tok])
            seen_tokens.add(tok)

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
    if isinstance(base, dict) and isinstance(customized, dict):
        merged: Dict[str, Any] = {}
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


def _insert_blank_lines_between_top_level_items(yaml_text: str) -> str:
    lines = yaml_text.splitlines()
    new_lines: List[str] = []
    for idx, line in enumerate(lines):
        is_top_level = (not line.startswith(" ")) and line.strip() != ""
        if is_top_level and new_lines:
            if new_lines and new_lines[-1] != "":
                new_lines.append("")
        new_lines.append(line)
    return "\n".join(new_lines) + ("\n" if not yaml_text.endswith("\n") else "")


def dump_yaml(data: Any, path: str) -> None:
    dumped = yaml.safe_dump(
        data,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
    )
    dumped = _insert_blank_lines_between_top_level_items(dumped)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumped)


def load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Merge two YAML files where customized overrides base on conflicts."
        )
    )
    parser.add_argument("-b", "--base", required=True, help="Path to base YAML file")
    parser.add_argument("-c", "--customized", required=True, help="Path to customized YAML file")
    parser.add_argument("-o", "--output", required=True, help="Path to write merged YAML output")
    args = parser.parse_args()

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
