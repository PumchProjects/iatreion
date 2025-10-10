import os
import tkinter as tk
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename

from iatreion.api import get_batched_result, get_result
from iatreion.configs import DataName, RrlEvalConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import get_config_path, load_dict, save_dict

names_mapping: dict[str, DataName] = {
    'MRI Volume': 'volume-pct-nz',
}

groups_list = [
    ['a', 'ac', 'l', 'g', 'gdn', 'o', 'm', 'ehikj'],
    ['1', '2', 'f'],
    ['LA', 'EA', 'LAc', 'EAc'],
    ['A1', 'A2', 'A3', 'A1c', 'A2c', 'A3c'],
]


def load_config(path: Path) -> RrlEvalConfig:
    config_dict = load_dict(path)
    return RrlEvalConfig(**config_dict.get('rrl-eval', {}))


def save_config(config: RrlEvalConfig, path: Path) -> None:
    config_dict = {'rrl-eval': asdict(config)}
    save_dict(config_dict, path)


def get_key[T, U](mapping: dict[T, U], value: U) -> T:
    for k, v in mapping.items():
        if v == value:
            return k
    return next(iter(mapping))


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
    dialog = create_dialog(master, 'Groups')
    frm = ttk.Frame(dialog, padding=5)
    frm.pack()
    vars: dict[str, tk.BooleanVar] = {}
    for column, groups in enumerate(groups_list):
        for row, group in enumerate(groups):
            var = tk.BooleanVar(value=group in init_groups)
            chk = ttk.Checkbutton(frm, text=group, variable=var)
            chk.grid(row=row, column=column, sticky=tk.W, padx=10)
            vars[group] = var
    ok_button = ttk.Button(dialog, text='OK', command=dialog.destroy)
    ok_button.pack(pady=5)
    master.wait_window(dialog)
    return [group for group, var in vars.items() if var.get()]


def save_batched_result(config: RrlEvalConfig) -> None:
    result = get_batched_result(config)
    path = asksaveasfilename(
        defaultextension='.xlsx', filetypes=[('Excel files', '*.xlsx')]
    )
    if path:
        result.to_excel(path)


def make_table(
    master: tk.Misc,
    row: int,
    column: int,
    data: list[list[str]],
    title: str,
    *headers: str,
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
        tree.insert('', 'end', values=items)
    tree.grid(row=0, column=0, sticky=tk.NSEW)
    if rule:
        vsb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky=tk.NS)


def show_result(master: tk.Tk, config: RrlEvalConfig) -> None:
    result_list, bias_list, support_list, oppose_list = get_result(config)
    dialog = create_dialog(master, 'Results')
    frm = ttk.Frame(dialog)
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)
    make_table(frm, 0, 0, result_list, 'Result', 'Label', 'Score')
    make_table(frm, 1, 0, bias_list, 'Initial Bias', 'Label', 'Score')
    make_table(
        frm, 0, 1, support_list, 'Supporting Rules', 'Label', 'Score', 'Rule', rule=True
    )
    make_table(
        frm, 1, 1, oppose_list, 'Opposing Rules', 'Label', 'Score', 'Rule', rule=True
    )
    close_button = ttk.Button(dialog, text='Close', command=dialog.destroy)
    close_button.pack(pady=5)
    master.wait_window(dialog)


def show_error_message(message: str) -> None:
    messagebox.showerror('Error', message)


def main() -> None:
    config_path = get_config_path()
    config = load_config(config_path)

    root = tk.Tk()
    root.title('Iatreion')

    frm = ttk.Frame(root, padding=(10, 10, 10, 5))
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)

    name = tk.StringVar(value=get_key(names_mapping, config.name))
    groups = tk.StringVar(value=config.groups)
    thesaurus = tk.StringVar(value=os.path.basename(config.thesaurus))
    data = tk.StringVar(value=os.path.basename(config.data))
    vmri = tk.StringVar(value=os.path.basename(config.vmri))
    batched = tk.BooleanVar(value=config.batched)

    def set_groups() -> None:
        selected_groups = select_groups(root, config.groups.split(','))
        config.groups = ','.join(selected_groups)
        groups.set(config.groups)

    def set_thesaurus_path() -> None:
        if path := askdirectory(initialdir=config.thesaurus):
            config.thesaurus = path
            thesaurus.set(os.path.basename(path))

    def set_data_path() -> None:
        path = askopenfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel files', '*.xlsx')],
            initialfile=config.data,
        )
        if path:
            config.data = path
            data.set(os.path.basename(path))

    def set_vmri_path() -> None:
        path = askopenfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel files', '*.xlsx')],
            initialfile=config.vmri,
        )
        if path:
            config.vmri = path
            vmri.set(os.path.basename(path))

    make_menu_row(frm, 0, 'Name:', name, *names_mapping.keys())
    make_row(frm, 1, 'Groups:', groups, 'Select', set_groups)
    make_row(frm, 2, 'Models:', thesaurus, 'Browse', set_thesaurus_path)
    make_row(frm, 3, 'Data:', data, 'Browse', set_data_path)
    make_row(frm, 4, 'Vmri:', vmri, 'Browse', set_vmri_path)

    def run_inference() -> None:
        config.name = names_mapping[name.get()]
        config.batched = batched.get()
        save_config(config, config_path)
        try:
            if config.batched:
                save_batched_result(config)
            else:
                show_result(root, config)
        except IatreionException as e:
            e.update(dataset=name.get(), groups=groups.get())
            show_error_message(str(e))
        except Exception as e:
            show_error_message(str(e))

    bottom_frm = ttk.Frame(root, padding=(10, 5, 10, 10))
    bottom_frm.pack()

    batch_chk = ttk.Checkbutton(bottom_frm, text='Batch Inference', variable=batched)
    batch_chk.pack(side=tk.LEFT, padx=5)

    run_btn = ttk.Button(bottom_frm, text='Run', command=run_inference)
    run_btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()
