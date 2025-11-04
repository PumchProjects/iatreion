import os
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Literal, Self

from iatreion.configs import DataName, RrlEvalConfig

from .static import groups_mapping, keep_mapping, names_mapping


def get_key[T, U](mapping: dict[T, U], value: U) -> T:
    for k, v in mapping.items():
        if v == value:
            return k
    return next(iter(mapping))


@dataclass
class ConfigBundle:
    config: RrlEvalConfig
    names: tk.StringVar = field(default_factory=tk.StringVar)
    groups: tk.StringVar = field(default_factory=tk.StringVar)
    thesaurus: tk.StringVar = field(default_factory=tk.StringVar)
    process: tk.StringVar = field(default_factory=tk.StringVar)
    data: defaultdict[str, tk.StringVar] = field(
        default_factory=lambda: defaultdict(tk.StringVar)
    )
    vmri: tk.StringVar = field(default_factory=tk.StringVar)
    change: tk.StringVar = field(default_factory=tk.StringVar)
    mode: tk.StringVar = field(default_factory=tk.StringVar)
    keep: tk.StringVar = field(default_factory=tk.StringVar)
    suspected: tk.BooleanVar = field(default_factory=tk.BooleanVar)
    debug: tk.BooleanVar = field(default_factory=tk.BooleanVar)

    def set_field(self, field: str) -> None:
        match field:
            case 'names':
                self.names.set(
                    ', '.join(names_mapping[name] for name in self.config.names)
                )
            case 'groups':
                self.groups.set(
                    ' : '.join(groups_mapping[group] for group in self.config.groups)
                )
            case 'thesaurus':
                self.thesaurus.set(os.path.basename(self.config.thesaurus))
            case 'process':
                self.process.set(os.path.basename(self.config.process))
            case 'data':
                for key, value in self.config.data.items():
                    self.data[key].set(os.path.basename(value))
            case 'vmri':
                self.vmri.set(os.path.basename(self.config.vmri))
            case 'change':
                self.change.set(os.path.basename(self.config.vmri_change))
            case 'mode':
                self.mode.set(self.config.mode)
            case 'keep':
                self.keep.set(keep_mapping[self.config.keep])
            case 'suspected':
                self.suspected.set(self.config.suspected_case)
            case 'debug':
                self.debug.set(self.config.debug)
            case _:
                raise ValueError(f'Unknown field: {field}')

    def set_names(self, names: list[DataName]) -> None:
        self.config.names = names
        self.set_field('names')

    def set_groups(self, groups: list[str]) -> None:
        self.config.groups = groups
        self.set_field('groups')

    def set_thesaurus(self, path: str) -> None:
        self.config.thesaurus = path
        self.set_field('thesaurus')

    def set_process(self, path: str) -> None:
        self.config.process = path
        self.set_field('process')

    def set_data(self, data: dict[str, str]) -> None:
        self.config.data |= data
        self.set_field('data')

    def set_vmri(self, path: str) -> None:
        self.config.vmri = path
        self.set_field('vmri')

    def set_vmri_change(self, path: str) -> None:
        self.config.vmri_change = path
        self.set_field('change')

    def set_mode(self, mode: Literal['single', 'batch', 'eval', 'show']) -> None:
        self.config.mode = mode
        self.set_field('mode')

    def set_keep(self, keep: str) -> None:
        self.config.keep = get_key(keep_mapping, keep)

    def set_suspected(self) -> None:
        self.config.suspected_case = self.suspected.get()

    def set_debug(self) -> None:
        self.config.debug = self.debug.get()

    @classmethod
    def from_config(cls, config: RrlEvalConfig) -> Self:
        bundle = cls(config=config)
        for f in fields(cls):
            if f.name != 'config':
                bundle.set_field(f.name)
        return bundle
