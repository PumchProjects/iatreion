import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.figure import Figure
from tableone import TableOne

from iatreion.configs import ShowConfig, ShowDataConfig
from iatreion.exceptions import IatreionException
from iatreion.train_utils import merge_data, read_data

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
column_name_mapping: dict[str, str] = {
    '年龄': 'Age',
    '性别': 'Sex',
    '教育年限': 'Education',
    'MMSE_总分': 'MMSE',
}


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.rename(columns=column_name_mapping, inplace=True)
    return frame


def _load_data_with_features(
    config: ShowDataConfig,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    dataset, train = config.make_configs()
    X_dfs, y_dfs, _, f_dfs = read_data(dataset, train)
    X_df, y_df, f_df = merge_data(X_dfs, y_dfs, f_dfs)
    y_df = y_df.map(group_mapping)
    X_df = _rename_columns(X_df)
    f_df = f_df.copy()
    f_df['name'] = f_df['name'].replace(column_name_mapping)
    data = pd.concat([X_df, y_df.to_frame('Label')], axis=1).convert_dtypes()
    groups = [group_mapping[group] for group in train._group_names]
    return data, groups, f_df


def _add_mmse_subscores(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    mmse_domains = {
        'Orientation': ['MMSE_时间定向', 'MMSE_地点定向'],
        'Registration': ['MMSE_瞬时记忆'],
        'Attention & Calculation': ['MMSE_计算功能'],
        'Recall': ['MMSE_延迟记忆'],
        'Language': ['MMSE_执行功能', 'MMSE_语言功能', 'MMSE_视空间功能'],
    }
    for output_name, input_columns in mmse_domains.items():
        if all(column in data.columns for column in input_columns):
            data[output_name] = data[input_columns].sum(
                axis=1,
                min_count=len(input_columns),
            )
    return data


def get_data_with_features(
    config: ShowDataConfig,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    data, groups, f_df = _load_data_with_features(config)
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'女': 'Female', '男': 'Male', pd.NA: 'Unknown'})
    data = _add_mmse_subscores(data)
    return data, groups, f_df


def _require_columns(data: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise IatreionException(
            'The "$context" view requires columns: $columns.',
            context=context,
            columns=', '.join(missing),
        )


def get_data(config: ShowDataConfig) -> tuple[pd.DataFrame, list[str]]:
    data, groups, _ = get_data_with_features(config)
    return data, groups


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    if pvalues.empty:
        return pvalues.copy()

    values = pvalues.to_numpy(dtype=float, copy=True)
    order = np.argsort(values)
    ranked = values[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty_like(adjusted)
    result[order] = adjusted
    return pd.Series(result, index=pvalues.index, dtype=float)


def _format_mean_std(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors='coerce').dropna()
    if numeric.empty:
        return 'NA'
    return f'{numeric.mean():.3f} +/- {numeric.std(ddof=1):.3f}'


def _get_candidate_features(data: pd.DataFrame, f_df: pd.DataFrame) -> list[str]:
    feature_names = f_df['name'].dropna().astype(str).tolist()
    seen: set[str] = set()
    ordered_names: list[str] = []
    for name in feature_names:
        if name in data.columns and name not in seen:
            ordered_names.append(name)
            seen.add(name)
    return ordered_names


def make_feature_difference_table(
    config: ShowDataConfig,
    *,
    top_k: int = 20,
    method: str = 'welch',
) -> pd.DataFrame:
    if method not in {'welch', 'mannwhitney'}:
        raise IatreionException(
            'Unsupported method "$method". Use "welch" or "mannwhitney".',
            method=method,
        )

    data, groups, f_df = _load_data_with_features(config)
    if len(groups) != 2:
        raise IatreionException(
            'Feature difference test requires exactly 2 groups, got $n.',
            n=str(len(groups)),
        )

    group_a, group_b = groups
    subset = data.loc[data['Label'].isin(groups)].copy()
    group_a_df = subset.loc[subset['Label'] == group_a]
    group_b_df = subset.loc[subset['Label'] == group_b]

    rows: list[dict[str, float | int | str]] = []
    for feature in _get_candidate_features(subset, f_df):
        a = pd.to_numeric(group_a_df[feature], errors='coerce').dropna()
        b = pd.to_numeric(group_b_df[feature], errors='coerce').dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        if a.nunique() < 2 and b.nunique() < 2 and float(a.iloc[0]) == float(b.iloc[0]):
            continue

        if method == 'mannwhitney':
            stat, pvalue = stats.mannwhitneyu(a, b, alternative='two-sided')
            statistic_name = 'U'
        else:
            stat, pvalue = stats.ttest_ind(
                a,
                b,
                equal_var=False,
                nan_policy='omit',
            )
            statistic_name = 't'
        if np.isnan(stat) or np.isnan(pvalue):
            continue
        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        effect_size = (
            0.0
            if pooled_std == 0 or np.isnan(pooled_std)
            else ((a.mean() - b.mean()) / pooled_std)
        )
        rows.append(
            {
                'Feature': feature,
                f'{group_a} (n)': int(len(a)),
                f'{group_a} Mean +/- SD': _format_mean_std(a),
                f'{group_b} (n)': int(len(b)),
                f'{group_b} Mean +/- SD': _format_mean_std(b),
                'Mean Difference': float(a.mean() - b.mean()),
                statistic_name: float(stat),
                'P-value': float(pvalue),
                "Cohen's d": float(effect_size),
            }
        )

    if not rows:
        raise IatreionException(
            'No numeric features with enough samples were available for comparison.'
        )

    table = pd.DataFrame(rows)
    table['FDR q-value'] = _benjamini_hochberg(table['P-value'])
    table['Abs Mean Difference'] = table['Mean Difference'].abs()
    table.sort_values(
        by=['FDR q-value', 'P-value', 'Abs Mean Difference'],
        ascending=[True, True, False],
        inplace=True,
    )
    if top_k > 0:
        table = table.head(top_k)

    statistic_col = 'U' if method == 'mannwhitney' else 't'
    table = table.assign(
        **{
            'Mean Difference': table['Mean Difference'].map(
                lambda value: f'{value:+.3f}'
            ),
            statistic_col: table[statistic_col].map(lambda value: f'{value:.3f}'),
            'P-value': table['P-value'].map(lambda value: f'{value:.3e}'),
            'FDR q-value': table['FDR q-value'].map(lambda value: f'{value:.3e}'),
            "Cohen's d": table["Cohen's d"].map(lambda value: f'{value:+.3f}'),
        }
    )
    return table.drop(columns='Abs Mean Difference')


def make_table_1(config: ShowDataConfig) -> TableOne:
    data, groups = get_data(config)
    _require_columns(
        data,
        ['Age', 'Sex', 'Education', 'MMSE', 'Label'],
        context='table_1',
    )
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


def violin(config: ShowDataConfig, name: str, title: str | None = None) -> Figure:
    title = title or name
    data, groups = get_data(config)
    _require_columns(data, ['Label', name, 'Sex'], context=f'violin({name})')
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
    config: ShowDataConfig, name: str, categories: list[str], title: str | None = None
) -> Figure:
    title = title or name
    data, groups = get_data(config)
    _require_columns(data, ['Label', name], context=f'bar({name})')
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


def _radar(df: pd.DataFrame) -> Figure:
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


def radar(config: ShowDataConfig, domains: list[str]) -> Figure:
    data, groups = get_data(config)
    _require_columns(data, ['Label', *domains], context='radar')
    df = data.groupby('Label')[domains].mean()
    df = df.reindex(index=groups)
    df_max, df_min = df.max(), df.min()
    df = (df - df_min) / (df_max - df_min)
    fig = _radar(df)
    return fig


def save(
    config: ShowConfig,
    table: TableOne | pd.DataFrame | None = None,
    fig: Figure | None = None,
    **kw,
) -> str:
    if fig is not None:
        fig.savefig(config.get_output_path('png'), dpi=300)
    elif isinstance(table, TableOne):
        table.to_latex(config.get_output_path('tex'), escape=True)
    elif isinstance(table, pd.DataFrame):
        table.to_latex(config.get_output_path('tex'), index=False, escape=True)
    return table.to_string(**kw) if isinstance(table, pd.DataFrame) else str(table)
