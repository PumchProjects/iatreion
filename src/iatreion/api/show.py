import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from tableone import TableOne

from iatreion.configs import ShowConfig
from iatreion.rrl import merge_data, read_data

group_mapping: dict[str, str] = {
    'abc': 'AD + AD-mix + AD-like',
    'a': 'AD',
    'b': 'AD-like (A+ T-)',
    'c': 'AD-mix',
    'ac': 'AD + AD-mix',
    'deghijklmnop': 'Dementia (non-AD)',
    'defghijklmnopq': 'non-AD',
    'l': 'Clin-AD & bio-non-AD',
    'f': 'HC',
    'dgn': 'FTLD',
    'djn': "Parkinson's disease",
    'o': 'VAD',
    '1': 'Aβ+',
    '2': 'Aβ-',
}
color_mapping: dict[str, str] = {'Female': '#ff7fa7', 'Male': '#7fc4fc'}


def get_data(config: ShowConfig) -> tuple[pd.DataFrame, list[str]]:
    dataset, train = config.make_configs()
    X_dfs, y_dfs, _, f_dfs = read_data(dataset, train)
    X_df, y_df, _ = merge_data(X_dfs, y_dfs, f_dfs)
    y_df = y_df.map(group_mapping)
    data = pd.concat([X_df, y_df.to_frame('Label')], axis=1).convert_dtypes()
    data.rename(
        columns={
            '年龄': 'Age',
            '性别': 'Sex',
            '教育年限': 'Education',
            'MMSE_总分': 'MMSE',
        },
        inplace=True,
    )
    data['Sex'] = data['Sex'].map({'女': 'Female', '男': 'Male', pd.NA: 'Unknown'})
    data['Orientation'] = data['MMSE_时间定向'] + data['MMSE_地点定向']
    data['Registration'] = data['MMSE_瞬时记忆']
    data['Attention & Calculation'] = data['MMSE_计算功能']
    data['Recall'] = data['MMSE_延迟记忆']
    data['Language'] = (
        data['MMSE_执行功能'] + data['MMSE_语言功能'] + data['MMSE_视空间功能']
    )
    groups = [group_mapping[group] for group in train._group_names]
    return data, groups


def make_table_1(config: ShowConfig) -> TableOne:
    data, groups = get_data(config)
    table = TableOne(
        data,
        columns=['Age', 'Sex', 'Education', 'MMSE', 'Label'],
        groupby='Label',
        nonnormal=['Education', 'MMSE'],
        pval=True,
        htest_name=True,
        rename={
            'Age': 'Age (years)',
            'Education': 'Education (years)',
            'MMSE': 'MMSE Score',
        },
        order={'Sex': ['Female', 'Male', 'Unknown'], 'Label': groups},
        decimals={'Education': 0, 'MMSE': 0},
        dip_test=True,
        normal_test=True,
        tukey_test=True,
    )
    return table


def violin(config: ShowConfig, name: str, title: str | None = None) -> Figure:
    title = title or name
    data, groups = get_data(config)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    sns.violinplot(
        data,
        x='Label',
        y=name,
        hue='Sex',
        order=groups,
        hue_order=['Female', 'Male'],
        palette=color_mapping,
        inner='quart',
        split=True,
        density_norm='width',
        ax=ax,
    )
    ax.set(
        xlabel='Label', ylabel=title, title=f'Violin Plot of {title} by Label and Sex'
    )
    return fig


def bar(
    config: ShowConfig, name: str, categories: list[str], title: str | None = None
) -> Figure:
    title = title or name
    data, groups = get_data(config)
    df = pd.crosstab(data['Label'], data[name])
    df = df.reindex(index=groups, columns=categories)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    df.plot.bar(
        ax=ax,
        title=f'Stacked Bar Chart of {title} by Label',
        ylabel='Number of Participants',
        rot=0,
        stacked=True,
        color=color_mapping,
    )
    return fig


def radar_chart(df: pd.DataFrame) -> Figure:
    categories = list(df)
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(14, 10), layout='constrained')
    ax = fig.add_subplot(polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1], categories, size=14)

    ax.set_rlabel_position(0.5)
    ax.set_yticks([0, 0.5, 1.0], ['0', '0.5', '1.0'], color='grey', size=7)
    ax.set_ylim(-0.1, 1.0)

    for row in df.itertuples():
        values = list(row[1:])
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=row.Index)
        ax.fill(angles, values, alpha=0.1)

    legend = fig.legend(loc='upper right')
    for text in legend.get_texts():
        text.set_fontsize(14)

    xticks = ax.xaxis.get_major_ticks()
    for tick in xticks[1::3]:
        tick.set_pad(30)

    return fig


def radar(config: ShowConfig, domains: list[str]) -> Figure:
    data, groups = get_data(config)
    df = data.groupby('Label')[domains].mean()
    df = df.reindex(index=groups)
    df_max, df_min = df.max(), df.min()
    df = (df - df_min) / (df_max - df_min)
    fig = radar_chart(df)
    return fig
