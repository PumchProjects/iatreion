from dataclasses import asdict, dataclass, field
from typing import Any, Self


@dataclass
class ColumnInfo:
    code_map: dict[str, str] = field(default_factory=dict)
    min: float = 0.0
    max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class ProcessInfo:
    columns: list[str] = field(default_factory=list)
    column_info: dict[str, ColumnInfo] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        columns = data.get('columns', [])
        column_info: dict[str, ColumnInfo] = {}
        for col, info in data.get('column_info', {}).items():
            column_info[col] = ColumnInfo(**info)
        return cls(columns, column_info)

    def to_dict(self) -> dict[str, Any]:
        process_info: dict[str, Any] = {}
        if self.columns:
            process_info['columns'] = self.columns
        if self.column_info:
            process_info['column_info'] = {
                col: info.to_dict() for col, info in self.column_info.items()
            }
        return process_info

    def __getitem__(self, col: str) -> ColumnInfo:
        if col not in self.column_info:
            self.column_info[col] = ColumnInfo()
        return self.column_info[col]
