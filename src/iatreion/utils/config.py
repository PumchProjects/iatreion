from collections import defaultdict
from dataclasses import replace
from typing import Any


def apply_overrides[T: Any](obj: T, overrides: dict[str, Any]) -> T:
    direct: dict[str, Any] = {}
    nested: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for key, value in overrides.items():
        head, _, tail = key.partition('.')
        if not tail:
            direct[head] = value
            continue
        nested[head][tail] = value

    for head, child_overrides in nested.items():
        direct[head] = apply_overrides(getattr(obj, head), child_overrides)
    return replace(obj, **direct)
