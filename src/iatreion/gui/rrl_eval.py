import os
import tkinter as tk
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename

import tomli
import tomli_w

from iatreion.api import get_batched_result, get_result
from iatreion.configs import RrlEvalConfig
from iatreion.utils import get_config_path


def load_config(path: Path) -> RrlEvalConfig:
    if path.is_file():
        with path.open('rb') as f:
            config_dict = tomli.load(f)
        return RrlEvalConfig(**config_dict['rrl-eval'])
    else:
        return RrlEvalConfig()


def save_config(config: RrlEvalConfig, path: Path) -> None:
    config_dict = {'rrl-eval': asdict(config)}
    with path.open('wb') as f:
        tomli_w.dump(config_dict, f)


def make_row(
    master: tk.Misc,
    row: int,
    label_text: str,
    variable: tk.StringVar,
    command: Callable[[], None] | None = None,
) -> None:
    label = ttk.Label(master, text=label_text)
    label.grid(row=row, column=0, sticky=tk.EW)
    entry = ttk.Entry(master, textvariable=variable)
    entry.grid(row=row, column=1, sticky=tk.EW)
    if command is not None:
        button = ttk.Button(master, text='Browse', command=command)
        button.grid(row=row, column=2, sticky=tk.EW)


def create_dialog(master: tk.Tk, title: str) -> tk.Toplevel:
    dialog = tk.Toplevel(master)
    dialog.title(title)
    dialog.transient(master)
    dialog.grab_set()
    return dialog


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


def main() -> None:
    config_path = get_config_path()
    config = load_config(config_path)

    root = tk.Tk()
    root.title('Iatreion')

    frm = ttk.Frame(root, padding=(10, 10, 10, 5))
    frm.grid_columnconfigure(1, weight=1)
    frm.pack(fill=tk.X)

    name = tk.StringVar(value=config.name)
    groups = tk.StringVar(value=config.groups)
    thesaurus = tk.StringVar(value=os.path.basename(config.thesaurus))
    data = tk.StringVar(value=os.path.basename(config.data))
    vmri = tk.StringVar(value=os.path.basename(config.vmri))
    batched = tk.BooleanVar(value=config.batched)

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

    make_row(frm, 0, 'Name:', name)
    make_row(frm, 1, 'Groups:', groups)
    make_row(frm, 2, 'Models:', thesaurus, set_thesaurus_path)
    make_row(frm, 3, 'Data:', data, set_data_path)
    make_row(frm, 4, 'Vmri:', vmri, set_vmri_path)

    def run_inference() -> None:
        config.name = name.get()
        config.groups = groups.get()
        config.batched = batched.get()
        save_config(config, config_path)
        if config.batched:
            result = get_batched_result(config)
            path = asksaveasfilename(
                defaultextension='.xlsx', filetypes=[('Excel files', '*.xlsx')]
            )
            if path:
                result.to_excel(path)
        else:
            result_table, bias_table, support_table, oppose_table = get_result(config)
            dialog = create_dialog(root, 'Result')
            frm = ttk.Frame(dialog)
            frm.grid_columnconfigure(1, weight=1)
            frm.pack(fill=tk.X)
            make_table(frm, 0, 0, result_table, 'Result', 'Label', 'Score')
            make_table(frm, 1, 0, bias_table, 'Initial Bias', 'Label', 'Score')
            make_table(
                frm,
                0,
                1,
                support_table,
                'Supporting Rules',
                'Label',
                'Score',
                'Rule',
                rule=True,
            )
            make_table(
                frm,
                1,
                1,
                oppose_table,
                'Opposing Rules',
                'Label',
                'Score',
                'Rule',
                rule=True,
            )
            close_button = ttk.Button(dialog, text='Close', command=dialog.destroy)
            close_button.pack(pady=5)
            root.wait_window(dialog)

    bottom_frm = ttk.Frame(root, padding=(10, 5, 10, 10))
    bottom_frm.pack()

    batch_chk = ttk.Checkbutton(bottom_frm, text='Batch Inference', variable=batched)
    batch_chk.pack(side=tk.LEFT, padx=5)

    run_btn = ttk.Button(bottom_frm, text='Run', command=run_inference)
    run_btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()
