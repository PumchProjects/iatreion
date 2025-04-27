from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory, ExistingDirectory

type DataName = Literal['snp']


@Parameter(name='*')
@dataclass
class PreprocessorConfig:
    data_prefix: Annotated[ExistingDirectory, Parameter(name=['--prefix', '-p'])]
    'Prefix of the data files'

    data_name: Annotated[DataName, Parameter(name=['--name', '-n'])]
    'Name of the data file'

    group: Annotated[str, Parameter(name=['--group', '-g'])]
    'Group names of the experiment'

    output_prefix: Annotated[Directory, Parameter(name=['--output', '-o'])]
    'Prefix of the output files'
