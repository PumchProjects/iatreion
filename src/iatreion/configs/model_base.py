from dataclasses import dataclass
from typing import Annotated

from cyclopts import Parameter


@Parameter(name='*')
@dataclass(kw_only=True)
class ModelConfig:
    ith_kfold: Annotated[int, Parameter(parse=False)] = 0
