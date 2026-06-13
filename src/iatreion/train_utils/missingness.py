from dataclasses import dataclass


@dataclass(frozen=True)
class MissingnessFilterArtifact:
    selected_features: list[str]
    dropped_features: list[str]
    missing_rates: dict[str, float]
    observed_counts: dict[str, int]
    params: dict[str, object]
    version: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> 'MissingnessFilterArtifact':
        return cls(
            version=int(data['version']),
            selected_features=list(data['selected_features']),
            dropped_features=list(data['dropped_features']),
            missing_rates={
                str(name): float(value) for name, value in data['missing_rates'].items()
            },
            observed_counts={
                str(name): int(value) for name, value in data['observed_counts'].items()
            },
            params=dict(data['params']),
        )

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'selected_features': self.selected_features,
            'dropped_features': self.dropped_features,
            'missing_rates': self.missing_rates,
            'observed_counts': self.observed_counts,
            'params': self.params,
        }
