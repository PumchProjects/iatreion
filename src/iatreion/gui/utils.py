import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import cast

from iatreion.configs import DataName

from .static import groups_mapping, names_mapping


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


def make_menu(
    master: tk.Misc,
    variable: tk.StringVar,
    command: Callable[[str], None],
    *values: str,
) -> None:
    menu = ttk.OptionMenu(master, variable, variable.get(), *values, command=command)  # type: ignore
    menu.pack(side=tk.LEFT, padx=5)


def make_check(
    master: tk.Misc, text: str, variable: tk.BooleanVar, command: Callable[[], None]
) -> None:
    check_btn = ttk.Checkbutton(master, text=text, variable=variable, command=command)
    check_btn.pack(side=tk.LEFT, padx=5)


def make_button(master: tk.Misc, text: str, command: Callable[[], None]) -> None:
    button = ttk.Button(master, text=text, command=command)
    button.pack(side=tk.LEFT, padx=5)


def create_dialog(master: tk.Tk, title: str) -> tk.Toplevel:
    dialog = tk.Toplevel(master)
    dialog.title(title)
    dialog.transient(master)
    dialog.grab_set()
    return dialog


def select_items(
    master: tk.Tk,
    items_list: list[list[str]],
    init_items: list[str],
    title: str,
    item_name_mapping: dict[str, str] | None = None,
) -> list[str]:
    dialog = create_dialog(master, title)
    frm = ttk.Frame(dialog, padding=(10, 10, 10, 5))
    frm.pack()
    vars: dict[str, tk.BooleanVar] = {}
    for column, items in enumerate(items_list):
        for row, item in enumerate(items):
            var = tk.BooleanVar(value=item in init_items)
            text = item_name_mapping[item] if item_name_mapping else item
            chk = ttk.Checkbutton(frm, text=text, variable=var)
            chk.grid(row=row, column=column, sticky=tk.W, padx=10)
            vars[item] = var

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
    return [item for item, var in vars.items() if var.get()]


def make_table(
    master: tk.Misc,
    row: int,
    column: int,
    data: list[list[str]],
    title: str,
    *headers: str,
) -> None:
    frm = ttk.Labelframe(master, text=title)
    frm.grid_columnconfigure(0, weight=1)
    frm.grid(row=row, column=column, sticky=tk.EW, padx=5, pady=5)
    tree = ttk.Treeview(frm, columns=headers, show='headings')
    scroll = False
    for header in headers:
        tree.heading(header, text=header)
        if header == '规则':
            tree.column(header, width=400, stretch=True, anchor=tk.W)
            scroll = True
        else:
            tree.column(header, width=100, stretch=False, anchor=tk.CENTER)
    for items in data:
        for i in range(len(headers)):
            if headers[i] == '模块':
                items[i] = names_mapping[cast(DataName, items[i])]
            elif headers[i] == '分组':
                items[i] = groups_mapping.get(items[i], '失败')
        tree.insert('', tk.END, values=items)
    tree.grid(row=0, column=0, sticky=tk.NSEW)
    if scroll:
        vsb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky=tk.NS)


def show_error_message(message: str) -> None:
    messagebox.showerror('错误', message)
