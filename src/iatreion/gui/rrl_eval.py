import os
import tkinter as tk
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename
from typing import Self, cast

from iatreion.api import get_batched_result, get_result
from iatreion.configs import DataName, RrlEvalConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import get_config_path, load_dict, save_dict

names_mapping: dict[DataName, str] = {
    'volume-new-pct': '核磁体积',
}

groups_list = [
    ['a', 'ac', 'l', 'gdn', 'o'],
    ['1', '2', 'f'],
    ['A<60', 'A<60C<60', 'F<60', 'A>60', 'A>60C>60', 'F>60'],
    ['A1', 'A1C1', 'F1', 'A2', 'A2C2', 'F2', 'A3', 'A3C3', 'F3'],
]


def load_config(path: Path) -> RrlEvalConfig:
    config_dict = load_dict(path)
    return RrlEvalConfig(**config_dict.get('rrl-eval', {}))


def save_config(config: RrlEvalConfig, path: Path) -> None:
    config_dict = {'rrl-eval': asdict(config)}
    save_dict(config_dict, path)


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
    vmri_change: tk.StringVar = field(default_factory=tk.StringVar)
    batched: tk.BooleanVar = field(default_factory=tk.BooleanVar)
    debug: tk.BooleanVar = field(default_factory=tk.BooleanVar)

    def set_names(self, names: list[DataName]) -> None:
        self.config.names = names
        self.names.set(', '.join(names_mapping[name] for name in names))

    def set_groups(self, groups: list[str]) -> None:
        self.config.groups = groups
        self.groups.set(', '.join(groups))

    def set_thesaurus(self, path: str) -> None:
        self.config.thesaurus = path
        self.thesaurus.set(os.path.basename(path))

    def set_process(self, path: str) -> None:
        self.config.process = path
        self.process.set(os.path.basename(path))

    def set_data(self, data: dict[str, str]) -> None:
        self.config.data = data
        for key, value in data.items():
            self.data[key].set(os.path.basename(value))

    def set_vmri(self, path: str) -> None:
        self.config.vmri = path
        self.vmri.set(os.path.basename(path))

    def set_vmri_change(self, path: str) -> None:
        self.config.vmri_change = path
        self.vmri_change.set(os.path.basename(path))

    def set_batched(self, batched: bool, *, init: bool = False) -> None:
        self.config.batched = batched
        if init:
            self.batched.set(batched)

    def set_debug(self, debug: bool, *, init: bool = False) -> None:
        self.config.debug = debug
        if init:
            self.debug.set(debug)

    @classmethod
    def from_config(cls, config: RrlEvalConfig) -> Self:
        bundle = cls(config=config)
        bundle.set_names(config.names)
        bundle.set_groups(config.groups)
        bundle.set_thesaurus(config.thesaurus)
        bundle.set_process(config.process)
        bundle.set_data(config.data)
        bundle.set_vmri(config.vmri)
        bundle.set_vmri_change(config.vmri_change)
        bundle.set_batched(config.batched, init=True)
        bundle.set_debug(config.debug, init=True)
        return bundle


def make_menu_row(
    master: tk.Misc,
    row: int,
    label_text: str,
    variable: tk.StringVar,
    *values: str,
) -> None:
    label = ttk.Label(master, text=label_text)
    label.grid(row=row, column=0, sticky=tk.EW)
    menu = ttk.OptionMenu(master, variable, variable.get(), *values)
    menu.grid(row=row, column=1, sticky=tk.EW)


def make_row(
    master: tk.Misc,
    row: int,
    label_text: str,
    variable: tk.StringVar,
    button_text: str = '',
    command: Callable[[], None] | None = None,
) -> None:
    label = ttk.Label(master, text=label_text)
    label.grid(row=row, column=0, sticky=tk.EW)
    entry = ttk.Entry(master, textvariable=variable)
    entry.grid(row=row, column=1, sticky=tk.EW)
    if command is not None:
        entry.config(state='readonly')
        button = ttk.Button(master, text=button_text, command=command)
        button.grid(row=row, column=2, sticky=tk.EW)


def create_dialog(master: tk.Tk, title: str) -> tk.Toplevel:
    dialog = tk.Toplevel(master)
    dialog.title(title)
    dialog.transient(master)
    dialog.grab_set()
    return dialog


def select_groups(master: tk.Tk, init_groups: list[str]) -> list[str]:
    dialog = create_dialog(master, '选择分组')
    frm = ttk.Frame(dialog, padding=(10, 10, 10, 5))
    frm.pack()
    vars: dict[str, tk.BooleanVar] = {}
    for column, groups in enumerate(groups_list):
        for row, group in enumerate(groups):
            var = tk.BooleanVar(value=group in init_groups)
            chk = ttk.Checkbutton(frm, text=group, variable=var)
            chk.grid(row=row, column=column, sticky=tk.W, padx=10)
            vars[group] = var

    def clear() -> None:
        for var in vars.values():
            var.set(False)

    bottom_frm = ttk.Frame(dialog, padding=(10, 5, 10, 10))
    bottom_frm.pack()
    clear_button = ttk.Button(bottom_frm, text='清空', command=clear)
    clear_button.pack(side=tk.LEFT, padx=5)
    ok_button = ttk.Button(bottom_frm, text='确定', command=dialog.destroy)
    ok_button.pack(side=tk.LEFT, padx=5)
    master.wait_window(dialog)
    return [group for group, var in vars.items() if var.get()]


def save_batched_result(config: RrlEvalConfig) -> None:
    result = get_batched_result(config)
    path = asksaveasfilename(
        defaultextension='.xlsx', filetypes=[('Excel 表格', '*.xlsx')]
    )
    if path:
        result.to_excel(path, float_format='%.4f')


def make_table(
    master: tk.Misc,
    row: int,
    column: int,
    data: list[list[str]],
    title: str,
    *headers: str,
    module: bool = True,
    rule: bool = False,
) -> None:
    frm = ttk.Labelframe(master, text=title)
    frm.grid_columnconfigure(0, weight=1)
    frm.grid(row=row, column=column, sticky=tk.EW, padx=5, pady=5)
    tree = ttk.Treeview(frm, columns=headers, show='headings')
    for idx, header in enumerate(headers):
        tree.heading(header, text=header)
        if rule and idx == len(headers) - 1:
            tree.column(header, width=400, stretch=True, anchor=tk.W)
        else:
            tree.column(header, width=100, stretch=False, anchor=tk.CENTER)
    for items in data:
        if module:
            items[0] = names_mapping[cast(DataName, items[0])]
        tree.insert('', 'end', values=items)
    tree.grid(row=0, column=0, sticky=tk.NSEW)
    if rule:
        vsb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky=tk.NS)


def show_result(master: tk.Tk, config: RrlEvalConfig) -> None:
    result_list, score_list, bias_list, support_list, oppose_list = get_result(config)
    dialog = create_dialog(master, '预测结果')
    frm = ttk.Frame(dialog)
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)
    make_table(
        frm, 0, 0, result_list, '最终结果', '分组', '分数', '置信度', module=False
    )
    make_table(
        frm,
        0,
        1,
        score_list,
        '各模块预测结果',
        '模块',
        '分组',
        '分数',
        '置信度',
        '权重',
    )
    make_table(frm, 1, 0, bias_list, '初始偏差', '模块', '分组', '分数')
    make_table(
        frm, 1, 1, support_list, '支持规则', '模块', '分组', '分数', '规则', rule=True
    )
    make_table(
        frm, 2, 1, oppose_list, '反对规则', '模块', '分组', '分数', '规则', rule=True
    )
    close_button = ttk.Button(dialog, text='关闭', command=dialog.destroy)
    close_button.pack(pady=5)
    master.wait_window(dialog)


def show_error_message(message: str) -> None:
    messagebox.showerror('错误', message)


def main() -> None:
    config_path = get_config_path()
    config = load_config(config_path)

    root = tk.Tk()
    root.title('Iatreion')

    frm = ttk.Frame(root, padding=(10, 10, 10, 5))
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)

    bundle = ConfigBundle.from_config(config)

    def set_groups() -> None:
        selected_groups = select_groups(root, config.groups)
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

    def set_data_path() -> None:
        path = askopenfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel 表格', '*.xlsx')],
            initialfile=config.data['volume-new'],
        )
        if path:
            bundle.set_data({'volume-new': path})

    def set_vmri_path() -> None:
        path = askopenfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel 表格', '*.xlsx')],
            initialfile=config.vmri,
        )
        if path:
            bundle.set_vmri(path)

    def set_vmri_change_path() -> None:
        path = askopenfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel 表格', '*.xlsx')],
            initialfile=config.vmri_change,
        )
        if path:
            bundle.set_vmri_change(path)

    def set_batched() -> None:
        bundle.set_batched(bundle.batched.get())

    def set_debug() -> None:
        bundle.set_debug(bundle.debug.get())

    make_menu_row(frm, 0, '模块:', bundle.names, *names_mapping.values())
    make_row(frm, 1, '分组:', bundle.groups, '选择分组', set_groups)
    make_row(frm, 2, '模型:', bundle.thesaurus, '选择文件夹', set_thesaurus_path)
    make_row(frm, 3, '预处理信息:', bundle.process, '选择文件', set_process_path)
    make_row(frm, 4, '数据:', bundle.data['volume-new'], '选择文件', set_data_path)
    make_row(frm, 5, '核磁体积均值标准差:', bundle.vmri, '选择文件', set_vmri_path)
    make_row(
        frm,
        6,
        '核磁体积表头变化:',
        bundle.vmri_change,
        '选择文件',
        set_vmri_change_path,
    )

    def run_inference() -> None:
        save_config(config, config_path)
        try:
            if config.batched:
                save_batched_result(config)
            else:
                show_result(root, config)
        except IatreionException as e:
            e.update(dataset=bundle.names.get(), groups=bundle.groups.get())
            show_error_message(str(e))
        except Exception as e:
            show_error_message(str(e))

    bottom_frm = ttk.Frame(root, padding=(10, 5, 10, 10))
    bottom_frm.pack()

    debug_chk = ttk.Checkbutton(
        bottom_frm, text='调试模式', variable=bundle.debug, command=set_debug
    )
    debug_chk.pack(side=tk.LEFT, padx=5)

    batch_chk = ttk.Checkbutton(
        bottom_frm, text='批量预测', variable=bundle.batched, command=set_batched
    )
    batch_chk.pack(side=tk.LEFT, padx=5)

    run_btn = ttk.Button(bottom_frm, text='运行', command=run_inference)
    run_btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()
