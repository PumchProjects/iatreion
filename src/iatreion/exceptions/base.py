from collections import defaultdict
from string import Template


class IatreionException(Exception):
    def __init__(self, template: str, **kwds: str) -> None:
        super().__init__()
        self.template = Template(template)
        self.mapping = defaultdict(str, kwds)

    def update(self, **kwds: str | None) -> None:
        for key, value in kwds.items():
            if value is not None:
                self.mapping[key] = value

    def __str__(self) -> str:
        return self.template.safe_substitute(self.mapping)
