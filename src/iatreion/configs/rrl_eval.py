from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory, ExistingFile

from .dataset import DataName, DatasetConfig
from .model_rrl_discrete import DiscreteRrlConfig
from .preprocessor import PreprocessorConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class RrlEvalConfig:
    name: Annotated[DataName, Parameter(name=['--name', '-n'])]
    'Name of the data file.'

    groups: Annotated[str, Parameter(name=['--groups', '-g'])]
    'Group names of the data.'

    thesaurus: Annotated[ExistingDirectory, Parameter(name=['--thesaurus', '-t'])]
    'Root directory for trained RRL models.'

    data: Annotated[ExistingFile, Parameter(name=['--data', '-d'])]
    'Path to the data file.'

    vmri: Annotated[ExistingFile, Parameter(name=['--vmri', '-v'])]
    'Path to the Vmri_mean_sd data file.'

    batched: Annotated[bool, Parameter(name=['--batched', '-b'], negative='')] = False
    'Whether to use batched inference.'

    def make_configs(self) -> tuple[PreprocessorConfig, DiscreteRrlConfig]:
        # HACK: Empty prefix
        dataset = DatasetConfig(prefix=Path(), name=self.name, simple=True)
        train = TrainConfig(group_names_=self.groups, final=True)
        # HACK: Empty output prefix
        process_config = PreprocessorConfig(
            dataset=dataset,
            output_prefix=Path(),
            vmri_data_path_=self.vmri,
            data_path_=self.data,
            final=True,
        )
        rrl_config = DiscreteRrlConfig(
            dataset=dataset, train=train, thesaurus=self.thesaurus
        )
        return process_config, rrl_config
