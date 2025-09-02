from string import Template


class IatreionException(Exception):
    def __init__(self, template: str, **kwds) -> None:
        super().__init__()
        self.template = Template(template)
        self.mapping = kwds

    def update(self, **kwds) -> None:
        self.mapping |= kwds

    def __str__(self) -> str:
        return self.template.safe_substitute(self.mapping)
