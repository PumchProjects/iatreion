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
    'h-demo': '人口学',
    'h-mmse': 'MMSE',
    'h-moca': 'MoCA',
    'h-mri': '核磁体积',
    'h-mri-roi': '核磁体积（分区）',
    'h-plasma': '血浆生物标记物',
    'h-history': '病史',
    'sh-apoe-labdata': 'APOE + 血化验',
}

data_mapping: dict[str, str] = {
    'history': '病史',
    'screen': '认知筛查',
    'composite': '认知综合',
    'biomarker': '血液生物标记物',
    'cbf': '核磁CBF',
    'csvd': '核磁CSVD',
    'volume-new': '核磁体积',
    'harmonized': '标准化',
}

names_list: list[list[str]] = [
    [
        'symptom',
        's-screen-sum',
        's-screen-sum-pct',
        'composite-bin',
        'biomarker',
        'cbf',
        'csvd',
        'volume-new-pct',
    ],
    [
        'h-demo',
        'h-mmse',
        'h-moca',
        'h-mri',
        'h-mri-roi',
        'h-plasma',
        'h-history',
        'sh-apoe-labdata',
    ],
]

groups_list: list[list[str]] = [
    ['a', 'c', '@ac', '@abc', 'l', '@dgn', 'o', '@deghijklmnop', '@defghijklmnopq'],
    ['1', '2', 'f'],
    ['A+', 'A-', 'T+', 'T-', 'fast', 'slow'],
]

groups_mapping = {
    '@abc': 'AD + AD-mix + AD-like',
    'abc': 'AD + AD-mix + AD-like',
    'a': 'AD',
    'b': 'AD-like (A+ T-)',
    'c': 'AD-mix',
    '@ac': 'AD + AD-mix',
    'ac': 'AD + AD-mix',
    '@deghijklmnop': 'AD 外的其它痴呆',
    'deghijklmnop': 'AD 外的其它痴呆',
    '@defghijklmnopq': '其它',
    'defghijklmnopq': '其它',
    'l': 'Clin-AD & bio-nonAD',
    'f': 'HC',
    '@dgn': 'FTLD',
    'dgn': 'FTLD',
    'o': 'VAD',
    '1': 'Aβ+',
    '2': 'Aβ-',
    'A+': 'A+',
    'A-': 'A-',
    'T+': 'T+',
    'T-': 'T-',
    'fast': '快进展',
    'slow': '慢进展',
}

keep_mapping: dict[Literal['first', 'last'], str] = {
    'first': '保留第一条记录',
    'last': '保留最后一条记录',
}
