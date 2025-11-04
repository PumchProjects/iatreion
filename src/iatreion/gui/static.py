from typing import Literal

from iatreion.configs import DataName

names_mapping: dict[DataName, str] = {
    'symptom': '病史',
    's-screen-sum': '认知筛查',
    's-screen-sum-pct': '认知筛查（子项占比）',
    'composite-bin': '认知综合',
    'biomarker': '血液生物标记物',
    'cbf': '核磁CBF',
    'csvd': '核磁CSVD',
    'volume-new-pct': '核磁体积',
}

data_mapping: dict[str, str] = {
    'history': '病史',
    'screen': '认知筛查',
    'composite': '认知综合',
    'biomarker': '血液生物标记物',
    'cbf': '核磁CBF',
    'csvd': '核磁CSVD',
    'volume-new': '核磁体积',
}

names_list: list[list[str]] = [
    ['symptom'],
    ['s-screen-sum', 's-screen-sum-pct', 'composite-bin'],
    ['biomarker'],
    ['cbf', 'csvd', 'volume-new-pct'],
]

groups_list: list[list[str]] = [
    ['a', 'ac', 'abc', 'l', 'dgn', 'o', 'deghijklmnop', 'defghijklmnopq'],
    ['1', '2', 'f'],
    ['A<60', 'A<60C<60', 'F<60', 'A>60', 'A>60C>60', 'F>60'],
]

groups_mapping = {
    'abc': 'AD + AD-mix + AD-like',
    'a': 'AD',
    'b': 'AD-like (A+ T-)',
    'c': 'AD-mix',
    'ac': 'AD + AD-mix',
    'deghijklmnop': 'AD 外的其它痴呆',
    'defghijklmnopq': '其它',
    'l': 'Clin-AD & bio-nonAD',
    'f': 'HC',
    'dgn': 'FTLD',
    'o': 'VAD',
    'A<60': 'EOAD (<60)',
    'A>60': 'LOAD (>60)',
    'F<60': 'HC (<60)',
    'F>60': 'HC (>60)',
    'A<60C<60': 'AD + AD-mix (<60)',
    'A>60C>60': 'AD + AD-mix (>60)',
    '1': 'Aβ+',
    '2': 'Aβ-',
}

keep_mapping: dict[Literal['all', 'first', 'last'], str] = {
    'all': '保留所有记录',
    'first': '保留第一条记录',
    'last': '保留最后一条记录',
}
