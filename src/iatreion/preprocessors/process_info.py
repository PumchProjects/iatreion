from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast

from iatreion.exceptions import IatreionException


@dataclass
class ProcessInfo:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    _: KW_ONLY
    final: bool

    def get_map(self, *keys: str) -> dict[str, Any]:
        map = self.attributes
        for key in keys:
            map = cast(dict[str, Any], map.setdefault(key, {}))
        return map

    def __getitem__(self, key: str | tuple[str, ...]) -> Any:
        if isinstance(key, str):
            key = (key,)
        map = self.get_map(*key[:-1])
        if key[-1] not in map:
            raise IatreionException(
                'No processing info "$keys" found for "$dataset"',
                keys='.'.join(key),
                dataset=self.name,
            )
        return map[key[-1]]

    def __call__[T](self, default_factory: Callable[[], T], *keys: str) -> T:
        if self.final:
            return cast(T, self[*keys])
        else:
            map = self.get_map(*keys[:-1])
            return cast(T, map.setdefault(keys[-1], default_factory()))

    def __setitem__(self, key: str | tuple[str, ...], value: Any) -> None:
        if isinstance(key, str):
            key = (key,)
        self.get_map(*key[:-1])[key[-1]] = value
