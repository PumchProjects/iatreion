import tkinter as tk
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename
from typing import Literal, cast

from iatreion.api import get_batched_result, get_eval_result, get_models, get_result
from iatreion.configs import DataName, RrlEvalConfig, name_data_mapping
from iatreion.exceptions import IatreionException
from iatreion.utils import get_config_path, load_dict, save_dict

from .bundle import ConfigBundle
from .static import (
    data_mapping,
    groups_list,
    groups_mapping,
    keep_mapping,
    names_list,
    names_mapping,
)
from .utils import (
    create_dialog,
    make_button,
    make_check,
    make_menu,
    make_row,
    make_table,
    select_items,
    show_error_message,
)


def load_config(path: Path) -> tuple[RrlEvalConfig, ConfigBundle]:
    config_dict = load_dict(path)
    try:
        config = RrlEvalConfig(**config_dict.get('rrl-eval', {}))
        bundle = ConfigBundle.from_config(config)
    except Exception:
        config = RrlEvalConfig()
        bundle = ConfigBundle.from_config(config)
    return config, bundle


def save_config(config: RrlEvalConfig, path: Path) -> None:
    config_dict = {'rrl-eval': asdict(config)}
    save_dict(config_dict, path)


def save_batched_result(config: RrlEvalConfig) -> None:
    result = get_batched_result(config)
    result.index.rename('ID', inplace=True)
    result.loc[:, 'Label'] = result['Label'].map(groups_mapping)
    path = asksaveasfilename(
        defaultextension='.xlsx', filetypes=[('Excel 表格', '*.xlsx')]
    )
    if path:
        result.to_excel(path, float_format='%.4f')


def show_result(master: tk.Tk, config: RrlEvalConfig) -> None:
    result_list, pred_list, bias_list, support_list, oppose_list = get_result(config)
    dialog = create_dialog(master, '预测结果')
    frm = ttk.Frame(dialog)
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)
    make_table(frm, 0, 0, result_list, '最终结果', '分组', '概率', '置信度')
    make_table(
        frm, 0, 1, pred_list, '各模块结果', '模块', '分组', '概率', '置信度', '权重'
    )
    make_table(frm, 1, 0, bias_list, '初始偏差', '模块', '分组', '分数')
    make_table(frm, 1, 1, support_list, '支持规则', '模块', '分组', '分数', '规则')
    make_table(frm, 2, 1, oppose_list, '反对规则', '模块', '分组', '分数', '规则')
    close_button = ttk.Button(dialog, text='关闭', command=dialog.destroy)
    close_button.pack(pady=5)
    master.wait_window(dialog)


def show_eval_result(master: tk.Tk, config: RrlEvalConfig) -> None:
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )

    result, fig, _ = get_eval_result(config)
    dialog = create_dialog(master, '预测结果')
    frm = ttk.Frame(dialog)
    frm.pack(fill=tk.BOTH, expand=True)
    left_frm = ttk.Frame(frm)
    left_frm.pack(side=tk.LEFT, fill=tk.Y)
    text_widget = tk.Text(left_frm, width=18, font=('Consolas', 20))
    text_widget.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)
    text_widget.insert(tk.END, result)
    text_widget.config(state='disabled')
    right_frm = ttk.Frame(frm)
    right_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    if fig is not None:
        canvas = FigureCanvasTkAgg(fig, master=right_frm)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, right_frm)
        toolbar.update()
        master.protocol('WM_DELETE_WINDOW', canvas.stop_event_loop)
    close_button = ttk.Button(dialog, text='关闭', command=dialog.destroy)
    close_button.pack(pady=5)
    master.wait_window(dialog)


def show_models(master: tk.Tk, config: RrlEvalConfig) -> None:
    rule_list = get_models(config)
    dialog = create_dialog(master, '查看模型')
    frm = ttk.Frame(dialog)
    frm.grid_columnconfigure(0, weight=1)
    frm.pack(fill=tk.X)
    data: list[list[str]] = []
    for name, rules in rule_list:
        rules[0].append('#初始偏差#')
        data += [[name, *rule] for rule in rules]
    make_table(frm, 0, 0, data, '', '模块', '分组', '分数', '规则')
    close_button = ttk.Button(dialog, text='关闭', command=dialog.destroy)
    close_button.pack(pady=5)
    master.wait_window(dialog)


def main() -> None:
    root = tk.Tk()
    root.title('Iatreion')

    frm = ttk.Frame(root, padding=(10, 10, 10, 5))
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)

    config_path = get_config_path()
    config, bundle = load_config(config_path)

    def set_data_path(data_name: str) -> Callable[[], None]:
        def set_data_path_inner() -> None:
            path = askopenfilename(
                defaultextension='.xlsx',
                filetypes=[('Excel 表格', '*.xlsx')],
                initialfile=config.data.get(data_name),
            )
            if path:
                bundle.set_data({data_name: path})

        return set_data_path_inner

    def make_data_rows() -> None:
        start = 6
        for widget in frm.grid_slaves():
            if int(widget.grid_info()['row']) >= start:
                widget.destroy()
        data_names = set(name_data_mapping[name] for name in config.names)
        for i, data_name in enumerate(data_names, start):
            label = f'{data_mapping[data_name]}数据:'
            command = set_data_path(data_name)
            make_row(frm, i, label, bundle.data[data_name], '选择文件', command)

    def set_names() -> None:
        selected_names = select_items(
            root,
            names_list,
            cast(list[str], config.names),
            '选择模块',
            item_name_mapping=cast(dict[str, str], names_mapping),
        )
        bundle.set_names([cast(DataName, name) for name in selected_names])
        make_data_rows()

    def set_groups() -> None:
        selected_groups = select_items(
            root,
            groups_list,
            config.groups,
            '选择分组',
            item_name_mapping=groups_mapping,
        )
        bundle.set_groups(selected_groups)

    def set_thesaurus_path() -> None:
        if path := askdirectory(initialdir=config.thesaurus):
            bundle.set_thesaurus(path)

    def set_process_path() -> None:
        path = askopenfilename(
            defaultextension='.toml',
            filetypes=[('TOML 文件', '*.toml')],
            initialfile=config.process,
        )
        if path:
            bundle.set_process(path)

    def set_vmri_path() -> None:
        path = askopenfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel 表格', '*.xlsx')],
            initialfile=config.vmri,
        )
        if path:
            bundle.set_vmri(path)

    def set_change_path() -> None:
        path = askopenfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel 表格', '*.xlsx')],
            initialfile=config.vmri_change,
        )
        if path:
            bundle.set_vmri_change(path)

    make_row(frm, 0, '模块:', bundle.names, '选择模块', set_names)
    make_row(frm, 1, '分组:', bundle.groups, '选择分组', set_groups)
    make_row(frm, 2, '模型:', bundle.thesaurus, '选择文件夹', set_thesaurus_path)
    make_row(frm, 3, '预处理信息:', bundle.process, '选择文件', set_process_path)
    make_row(frm, 4, '核磁体积均值标准差:', bundle.vmri, '选择文件', set_vmri_path)
    make_row(frm, 5, '核磁体积表头变化:', bundle.change, '选择文件', set_change_path)
    make_data_rows()

    def run_inference() -> None:
        save_config(config, config_path)
        try:
            match config.mode:
                case 'single':
                    show_result(root, config)
                case 'batch':
                    save_batched_result(config)
                case 'eval':
                    show_eval_result(root, config)
                case 'show':
                    show_models(root, config)
        except IatreionException as e:
            e.update(
                dataset=names_mapping.get(cast(DataName, e.mapping['dataset'])),
                groups=' : '.join(
                    groups_mapping.get(g, g) for g in e.mapping['groups'].split(', ')
                ),
                data_name=data_mapping.get(e.mapping['data_name']),
                vmri='核磁体积均值标准差',
                vmri_change='核磁体积表头变化',
                process_info='预处理信息',
            )
            show_error_message(str(e))
        except Exception as e:
            show_error_message(str(e))
            if config.debug:
                raise e

    def set_mode(
        mode: Literal['single', 'batch', 'eval', 'show'],
    ) -> Callable[[], None]:
        def callback() -> None:
            bundle.set_mode(mode)
            run_inference()

        return callback

    check_frm = ttk.Frame(root, padding=(10, 5, 10, 2))
    check_frm.pack()

    make_menu(check_frm, bundle.keep, bundle.set_keep, *keep_mapping.values())
    make_check(check_frm, '调试模式', bundle.debug, bundle.set_debug)
    make_check(check_frm, '疑似病例', bundle.suspected, bundle.set_suspected)

    button_frm = ttk.Frame(root, padding=(10, 2, 10, 10))
    button_frm.pack()

    make_button(button_frm, '查看模型', set_mode('show'))
    make_button(button_frm, '分析首例', set_mode('single'))
    make_button(button_frm, '批量预测', set_mode('eval'))
    make_button(button_frm, '批量导出', set_mode('batch'))

    root.mainloop()
