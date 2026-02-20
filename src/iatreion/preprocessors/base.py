from abc import ABC, abstractmethod
from collections.abc import Callable

import pandas as pd
from pandas.api.types import is_string_dtype

from iatreion.configs import DataName, PreprocessorConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import encode_string, logger, name_to_stem, stem_to_name

from .process_info import ProcessInfo


class Preprocessor(ABC):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__()
        self.config = config
        self.name = name
        self.data_name = config.get_data_name(name)
        self.stem_pattern = config.get_stem_pattern(self.data_name)
        self.process_info_: ProcessInfo | None = None

    @property
    def process_info(self) -> ProcessInfo:
        if self.process_info_ is None:
            if self.config.final:
                if self.name not in self.config.process_info_dict:
                    raise IatreionException(
                        'No processing info found for "$dataset"',
                        dataset=self.name,
                    )
                else:
                    info = self.config.process_info_dict[self.name]
                    self.process_info_ = ProcessInfo(self.name, info, final=True)
            else:
                self.process_info_ = ProcessInfo(self.name, final=False)
        return self.process_info_

    def save_process_info(self) -> None:
        if self.process_info_ is not None:
            info = self.process_info_.attributes
            self.config.process_info_dict[self.name] = info

    def get_group_names(self) -> pd.DataFrame:
        if self.config.contains_group_columns:
            return self.config.data[self.data_name][self.config.group_columns].copy()
        if 'group_names' not in self.config.data:
            data = pd.read_excel(
                self.config.group_data_path,
                index_col='serial_num',
                dtype_backend='numpy_nullable',
            )
            self.config.data['group_names'] = data[self.config.group_columns]
        return self.config.data['group_names'].copy()

    def merge_group_names(self, data: pd.DataFrame) -> pd.DataFrame:
        group_names = self.get_group_names()
        if self.config.contains_group_columns:
            data = pd.concat([data, group_names], axis=1)
        else:
            data = data.merge(
                group_names, how='left', left_index=True, right_index=True
            )
        return data

    def get_basic_data(self) -> pd.DataFrame:
        if 'basic_data' not in self.config.data:
            data = pd.read_excel(
                self.config.basic_data_path,
                index_col='serial_num',
                dtype_backend='numpy_nullable',
            )
            data.rename(columns={'实际出生日期': 'date of birth'}, inplace=True)
            self.config.data['basic_data'] = data
        return self.config.data['basic_data'].copy()

    def get_basic_info(
        self, data: pd.DataFrame, columns: list[str], *, force_final: bool = False
    ) -> pd.DataFrame:
        # HACK: Use the information included in the data when in final mode
        if self.config.final or force_final:
            for col in columns:
                if col not in data.columns:
                    raise IatreionException(
                        'Column "$column" not found in "$dataset".',
                        column=col,
                        dataset=self.name,
                    )
        else:
            basic_data = self.get_basic_data()
            data = data.merge(
                basic_data[columns],
                how='left',
                left_index=True,
                right_index=True,
                suffixes=('_unused', None),
            )
        return data

    def calc_ages(
        self, data: pd.DataFrame, date_col: str, *, force_final: bool = False
    ) -> tuple[pd.DataFrame, pd.Series]:
        data = self.get_basic_info(data, ['date of birth'], force_final=force_final)
        test_dates = pd.to_datetime(data[date_col], utc=True)
        birth_dates = pd.to_datetime(data['date of birth'], utc=True)
        real_ages = (test_dates - birth_dates).dt.days // 365.2422
        return data, real_ages

    def sum_columns(
        self,
        data: pd.DataFrame,
        columns: list[str],
        name: str,
        *,
        divisor: str | None = None,
    ) -> pd.DataFrame:
        # skipna=False ensures that NaN will propagate through the sum
        col: pd.Series = data[columns].sum(axis=1, skipna=False).astype('Int64')
        data = data.drop(columns=columns)
        data[name] = col / (data[divisor] + 1e-8) if divisor is not None else col
        return data

    def binarize_column(
        self,
        data: pd.DataFrame,
        column: str,
        threshold: int,
        name: str,
        ge_name: str,
        lt_name: str,
    ) -> pd.DataFrame:
        col: pd.Series = (data[column] >= threshold).astype('Int8')
        data = data.drop(columns=[column])
        data[name] = col.map({1: ge_name, 0: lt_name})
        return data

    def drop_columns(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        additional_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.config.final:
            if columns is not None:
                data = data.drop(columns=columns)
        else:
            data = data.drop(columns=(columns or []) + (additional_columns or []))
        return data

    def read_data(self) -> pd.DataFrame:
        if self.data_name not in self.config.data:
            data_path, sheet_name = self.config.get_data_path(self.data_name)
            data = pd.read_excel(
                data_path,
                sheet_name=sheet_name,
                # HACK: serial_num is needed for merging birth dates
                index_col=self.config.index_name,
                na_values=['/', '#NUM!'],
                dtype_backend='numpy_nullable',
            )
            self.config.data[self.data_name] = data
            if indices_names := self.config.get_indices_names(self.data_name):
                self.config.final_indices.append(data[indices_names].astype(str))
        return self.config.data[self.data_name].copy()

    @abstractmethod
    def get_data(self) -> pd.DataFrame: ...

    def get_data_outer(self) -> pd.DataFrame:
        data = self.get_data()
        if self.config.final:
            data.rename(columns=encode_string, inplace=True)
        else:
            self.save_process_info()
        return data

    def get_name_to_stem_callback(self) -> Callable[[str], str] | None:
        if self.stem_pattern is not None:
            return name_to_stem(self.stem_pattern)
        return None

    def get_stem_to_name_callback(self) -> Callable[[str], str] | None:
        if self.stem_pattern is not None:
            data = self.read_data()
            callback = name_to_stem(self.stem_pattern)
            mapping = {callback(name): name for name in data.columns}
            return stem_to_name(self.stem_pattern, mapping)
        return None

    @staticmethod
    def remove_useless_columns(data: pd.DataFrame) -> pd.DataFrame:
        nunique = data.nunique()
        columns = nunique[nunique <= 1].index
        if not columns.empty:
            logger.warning(
                f'[bold yellow]Removing useless columns:[/] {", ".join(columns)}',
                extra={'markup': True},
            )
            data = data.drop(columns=columns)
        return data

    def get_augmented_vector_name(self, data: pd.DataFrame) -> list[tuple[str, str]]:
        discrete_th = 10
        augmented_vector_name: list[tuple[str, str]] = []
        data = data.iloc[:, : -len(self.config.group_columns)]
        for name in data.columns:
            if is_string_dtype(data[name]):
                augmented_vector_name.append((name, 'unordered'))
            elif data[name].nunique() <= discrete_th:
                augmented_vector_name.append((name, 'ordered'))
            else:
                augmented_vector_name.append((name, 'continuous'))
        return augmented_vector_name

    def save_data(
        self, data: pd.DataFrame, augmented_vector_name: list[tuple[str, str]]
    ) -> None:
        feature_names = [f'{pair[0]} {pair[1]}\n' for pair in augmented_vector_name]
        with self.config.dataset.get_info(self.name).open('w', encoding='utf-8') as f:
            f.write(f'{self.config.index_name} index\n')
            f.writelines(feature_names)
            for col in self.config.group_columns:
                f.write(f'{col} label\n')
        fmap: list[str] = []
        for i, (name_, type_) in enumerate(augmented_vector_name):
            name = encode_string(name_, ' ')
            match type_:
                case 'unordered' | 'ordered':
                    fmap.append(f'{i}\t{name}\ti\n')
                case 'continuous':
                    fmap.append(f'{i}\t{name}\tq\n')
                case _:
                    raise ValueError(f'Unsupported type "{type_}" for "{name_}"')
        with self.config.dataset.get_fmap(self.name).open('w', encoding='utf-8') as f:
            f.writelines(fmap)
        data_file = self.config.dataset.get_data(self.name)
        data.to_csv(data_file, na_rep='<NA>', float_format='%.6f', header=False)

    def process(self) -> None:
        logger.info(
            f'[bold green]Processing "{self.name}" data...',
            extra={'markup': True},
        )
        data = self.get_data_outer()
        subset = data.columns
        data = self.merge_group_names(data)
        # Drop rows with less than 50% non-NaN values
        threshold = int(len(subset) * 0.5)
        data = data.dropna(axis=0, thresh=threshold, subset=subset)
        data = self.remove_useless_columns(data)
        augmented_vector_name = self.get_augmented_vector_name(data)
        logger.info('Saving data...')
        self.save_data(data, augmented_vector_name)
