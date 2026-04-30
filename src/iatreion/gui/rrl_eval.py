import tkinter as tk
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename
from typing import Literal, cast

from iatreion.api import (
    RrlTermOption,
    get_batched_result,
    get_eval_result,
    get_result,
    get_rule_options,
    get_rule_waterfall_data,
)
from iatreion.configs import DataName, RrlEvalConfig, name_data_mapping
from iatreion.exceptions import IatreionException
from iatreion.show_helpers import rrl_rule_waterfall_plot
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
    set_font,
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


def show_waterfall(master: tk.Tk, config: RrlEvalConfig) -> None:
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )

    bundle = get_rule_waterfall_data(config)
    fig = rrl_rule_waterfall_plot(bundle)
    dialog = create_dialog(master, f'RRL Waterfall - {bundle.sample.sample_id}')
    frm = ttk.Frame(dialog)
    frm.pack(fill=tk.BOTH, expand=True)
    fig_canvas = FigureCanvasTkAgg(fig, master=frm)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(fig_canvas, frm)
    toolbar.update()

    def close_dialog() -> None:
        dialog.destroy()

    ttk.Button(dialog, text='关闭', command=close_dialog).pack(pady=5)
    master.protocol('WM_DELETE_WINDOW', fig_canvas.stop_event_loop)
    master.wait_window(dialog)


def show_result(master: tk.Tk, config: RrlEvalConfig) -> None:
    (
        sample_id,
        result_list,
        pred_list,
        bias_list,
        support_list,
        oppose_list,
    ) = get_result(config)
    dialog = create_dialog(master, f'预测结果 - {sample_id}')
    frm = ttk.Frame(dialog)
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)
    make_table(
        frm,
        0,
        0,
        pred_list,
        '各模块结果',
        '模块',
        '分组',
        '分数',
        '概率',
        '权重',
    )
    make_table(
        frm,
        0,
        1,
        result_list,
        '最终结果',
        '分组',
        '分数',
        '边界',
        '概率',
        '阳性概率',
        '阈值',
    )
    make_table(frm, 1, 0, bias_list, '初始偏差', '模块', '分组', '分数')
    make_table(frm, 1, 1, support_list, '支持规则', '模块', '分组', '分数', '规则')
    make_table(frm, 2, 1, oppose_list, '反对规则', '模块', '分组', '分数', '规则')

    top_k = tk.StringVar(value=str(config.top_k))

    def show_result_waterfall() -> None:
        try:
            config.top_k = int(top_k.get() or 0)
            show_waterfall(master, config)
        except Exception as e:
            show_error_message(str(e))
            if config.debug:
                raise e
        if dialog.winfo_exists():
            dialog.grab_set()

    button_frm = ttk.Frame(dialog)
    button_frm.pack(pady=5)
    ttk.Label(button_frm, text='Waterfall规则数:').pack(side=tk.LEFT, padx=5)
    ttk.Entry(button_frm, textvariable=top_k, width=6).pack(side=tk.LEFT, padx=5)
    ttk.Button(
        button_frm,
        text='Waterfall',
        command=show_result_waterfall,
    ).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frm, text='关闭', command=dialog.destroy).pack(
        side=tk.LEFT,
        padx=5,
    )
    master.wait_window(dialog)


def show_eval_output(master: tk.Tk, config: RrlEvalConfig) -> None:
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
    text_widget = tk.Text(left_frm, width=36, font=('Consolas', 16))
    text_widget.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)
    text_widget.insert(tk.END, result)
    text_widget.config(state='disabled')
    right_frm = ttk.Frame(frm)
    right_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    if fig is not None:
        fig_canvas = FigureCanvasTkAgg(fig, master=right_frm)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(fig_canvas, right_frm)
        toolbar.update()
        master.protocol('WM_DELETE_WINDOW', fig_canvas.stop_event_loop)
    close_button = ttk.Button(dialog, text='关闭', command=dialog.destroy)
    close_button.pack(pady=5)
    master.wait_window(dialog)


def select_eval_terms(master: tk.Tk, config: RrlEvalConfig) -> None:
    options = get_rule_options(config)
    valid_modules = {option.module for option in options}
    config.enabled_biases = {
        module: enabled
        for module, enabled in config.enabled_biases.items()
        if module in valid_modules
    }
    config.enabled_rules = {
        module: indices
        for module, indices in config.enabled_rules.items()
        if module in valid_modules
    }
    dialog = create_dialog(master, '选择激活规则')

    body = ttk.Frame(dialog, padding=(10, 10, 10, 5))
    body.pack(fill=tk.BOTH, expand=True)

    headers = ('启用', '模块', '类型', '分组', '分数', '规则')
    tree = ttk.Treeview(body, columns=headers, show='headings', selectmode='browse')
    tree.tag_configure('disabled', foreground='gray')
    scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    for header in headers:
        tree.heading(header, text=header)
        match header:
            case '启用':
                tree.column(header, width=60, stretch=False, anchor=tk.CENTER)
            case '模块':
                tree.column(header, width=130, stretch=False, anchor=tk.CENTER)
            case '类型':
                tree.column(header, width=80, stretch=False, anchor=tk.CENTER)
            case '分组':
                tree.column(header, width=160, stretch=False, anchor=tk.CENTER)
            case '分数':
                tree.column(header, width=80, stretch=False, anchor=tk.CENTER)
            case '规则':
                tree.column(header, width=520, stretch=True, anchor=tk.W)

    def close_dialog() -> None:
        dialog.destroy()

    def selected_text(selected: bool) -> str:
        return '[x]' if selected else '[ ]'

    option_by_iid: dict[str, RrlTermOption] = {}
    selected_by_iid: dict[str, bool] = {}
    rule_counts: dict[str, int] = {}
    for option in options:
        module = option.module
        if option.kind == 'bias':
            selected = config.enabled_biases.get(module, True)
        else:
            selected_rules = config.enabled_rules.get(module)
            selected = selected_rules is None or option.index in selected_rules
            rule_counts[module] = rule_counts.get(module, 0) + 1

        iid = tree.insert(
            '',
            tk.END,
            values=(
                selected_text(selected),
                names_mapping.get(module, module),
                option.display_index,
                groups_mapping.get(option.label, option.label),
                f'{option.score:.2f}',
                option.rule,
            ),
        )
        option_by_iid[iid] = option
        selected_by_iid[iid] = selected
        if not selected:
            tree.item(iid, tags=('disabled',))

    def set_item_selected(iid: str, selected: bool) -> None:
        values = list(tree.item(iid, 'values'))
        values[0] = selected_text(selected)
        tree.item(iid, values=values, tags=() if selected else ('disabled',))
        selected_by_iid[iid] = selected

    def toggle_item(iid: str) -> None:
        set_item_selected(iid, not selected_by_iid[iid])

    def toggle_clicked_item(event: tk.Event) -> str | None:
        iid = tree.identify_row(event.y)
        if iid:
            toggle_item(iid)
            tree.selection_set(iid)
            tree.focus(iid)
            return 'break'
        return None

    def toggle_focused_item(_: tk.Event) -> str:
        iid = tree.focus()
        if iid:
            toggle_item(iid)
        return 'break'

    tree.bind('<ButtonRelease-1>', toggle_clicked_item)
    tree.bind('<space>', toggle_focused_item)

    bottom_frm = ttk.Frame(dialog, padding=(10, 5, 10, 10))
    bottom_frm.pack(fill=tk.X)

    fallback_var = tk.BooleanVar(value=config.zero_mean_fallback == 'bias')
    ttk.Checkbutton(
        bottom_frm,
        text='零分时使用 Bias 偏向',
        variable=fallback_var,
    ).pack(side=tk.LEFT, padx=5)

    def set_all(value: bool) -> None:
        for iid in selected_by_iid:
            set_item_selected(iid, value)

    def select_bias_only() -> None:
        for iid, option in option_by_iid.items():
            set_item_selected(iid, option.kind == 'bias')

    def disable_bias() -> None:
        for iid, option in option_by_iid.items():
            if option.kind == 'bias':
                set_item_selected(iid, False)

    def apply_selection() -> None:
        enabled_biases: dict[str, bool] = {}
        selected_rules: dict[str, list[int]] = {}
        for iid, option in option_by_iid.items():
            selected = selected_by_iid[iid]
            if option.kind == 'bias':
                if not selected:
                    enabled_biases[option.module] = False
            elif selected:
                assert option.index is not None
                selected_rules.setdefault(option.module, []).append(option.index)

        enabled_rules = {
            module: selected
            for module, total in rule_counts.items()
            if len(selected := selected_rules.get(module, [])) != total
        }
        config.enabled_biases = enabled_biases
        config.enabled_rules = enabled_rules
        config.zero_mean_fallback = 'bias' if fallback_var.get() else 'uniform'

    def show_selected_result() -> None:
        apply_selection()
        try:
            show_eval_output(master, config)
        except Exception as e:
            show_error_message(str(e))
            if config.debug:
                raise e
        if dialog.winfo_exists():
            dialog.grab_set()

    button_frm = ttk.Frame(bottom_frm)
    button_frm.pack(side=tk.RIGHT)
    ttk.Button(button_frm, text='全选', command=lambda: set_all(True)).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(button_frm, text='清空', command=lambda: set_all(False)).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(button_frm, text='只选 Bias', command=select_bias_only).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(button_frm, text='不选 Bias', command=disable_bias).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(button_frm, text='确定', command=show_selected_result).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(button_frm, text='关闭', command=close_dialog).pack(side=tk.LEFT, padx=5)

    master.wait_window(dialog)


def show_eval_result(master: tk.Tk, config: RrlEvalConfig) -> None:
    select_eval_terms(master, config)


def show_models(master: tk.Tk, config: RrlEvalConfig) -> None:
    options = get_rule_options(config)
    dialog = create_dialog(master, '查看模型')
    frm = ttk.Frame(dialog)
    frm.grid_columnconfigure(0, weight=1)
    frm.pack(fill=tk.X)
    data = [
        [
            option.module,
            option.display_index,
            option.label,
            f'{option.score:.2f}',
            option.rule,
        ]
        for option in options
    ]
    make_table(frm, 0, 0, data, '', '模块', '索引', '分组', '分数', '规则')
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
        row = start + len(data_names)
        make_row(frm, row, '患者ID列名:', bundle.index)
        make_row(frm, row + 1, '患者ID:', bundle.sample_id)
        make_row(frm, row + 2, '患者分组列名:', bundle.label)

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
        try:
            bundle.set_index()
            bundle.set_label()
            bundle.set_sample_id()
            save_config(config, config_path)
            set_font()
            match config.mode:
                case 'single':
                    show_result(root, config)
                case 'batch':
                    save_batched_result(config)
                case 'eval':
                    show_eval_result(root, config)
                    save_config(config, config_path)
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
                index_name='患者ID列名',
                label_name='患者分组列名',
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
    make_button(button_frm, '分析样本', set_mode('single'))
    make_button(button_frm, '批量预测', set_mode('eval'))
    make_button(button_frm, '批量导出', set_mode('batch'))

    root.mainloop()
