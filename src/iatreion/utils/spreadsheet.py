from pathlib import Path
from typing import Any

import pandas as pd

from iatreion.exceptions import IatreionException

EXCEL_SUFFIXES = {'.xlsx', '.xls', '.xlsm', '.xlsb', '.ods'}
CSV_SUFFIXES = {'.csv'}
TSV_SUFFIXES = {'.tsv', '.tab'}


def read_spreadsheet(path: str | Path, **kwds: Any) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        return pd.read_excel(path, **kwds)
    kwds.pop('sheet_name', None)
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(path, **kwds)
    if suffix in TSV_SUFFIXES:
        return pd.read_csv(path, sep='\t', **kwds)
    raise IatreionException(
        'Unsupported spreadsheet file "$path". Use xlsx, csv, or tsv.',
        path=str(path),
    )


def write_spreadsheet(path: str | Path, table: pd.DataFrame, **kwds: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        table.to_excel(path, **kwds)
    elif suffix in CSV_SUFFIXES:
        table.to_csv(path, **kwds)
    elif suffix in TSV_SUFFIXES:
        table.to_csv(path, sep='\t', **kwds)
    else:
        raise IatreionException(
            'Unsupported spreadsheet file "$path". Use xlsx, csv, or tsv.',
            path=str(path),
        )
    return path
