from dataclasses import dataclass
from pathlib import Path

from iatreion.utils import load_dict, save_dict

from .missingness import MissingnessFilterArtifact


@dataclass(frozen=True)
class RrlPreprocessingArtifact:
    available_columns: list[str]
    missing_aware_mode: str
    missingness_filter: MissingnessFilterArtifact | None = None
    version: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> 'RrlPreprocessingArtifact':
        missingness_filter = data.get('missingness_filter')
        return cls(
            version=int(data['version']),
            available_columns=list(data['available_columns']),
            missing_aware_mode=str(data['missing_aware_mode']),
            missingness_filter=(
                None
                if missingness_filter is None
                else MissingnessFilterArtifact.from_dict(missingness_filter)
            ),
        )

    @classmethod
    def load(cls, path: Path) -> 'RrlPreprocessingArtifact':
        return cls.from_dict(load_dict(path))

    def to_dict(self) -> dict:
        data = {
            'version': self.version,
            'available_columns': self.available_columns,
            'missing_aware_mode': self.missing_aware_mode,
        }
        if self.missingness_filter is not None:
            data['missingness_filter'] = self.missingness_filter.to_dict()
        return data

    def save(self, path: Path) -> None:
        save_dict(self.to_dict(), path)
