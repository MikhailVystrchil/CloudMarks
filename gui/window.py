import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from gui.core import PARAMS, Result, calculate


BG = "#eef2f7"
CARD = "#ffffff"
INK = "#17263c"
MUTED = "#66758a"
BLUE = "#2563eb"

SCAN_TYPES = [("Облака LAS / TXT", "*.las *.txt"), ("Все файлы", "*.*")]
REF_TYPES = [("Базовые точки", "*.txt *.csv *.tsv"), ("Все файлы", "*.*")]


def text(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


class Window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CloudMarks · анализ облаков точек")
        self.geometry("1440x880")
        self.minsize(1120, 720)
        self.configure(bg=BG)

        self.events = queue.Queue()
        self.result = Result(pd.DataFrame())
        self.view = pd.DataFrame()
        self.busy = False
        self.sort_reverse = {}

        self._style()
        self._build()
        self._mode_changed()
        self.after(100, self._poll)

    def _style(self):
        self.option_add("*Font", ("Segoe UI", 10))
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=INK)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Card.TLabel", background=CARD)
        style.configure("Hint.TLabel", background=CARD, foreground=MUTED)
        style.configure("Title.TLabel", background=CARD,
                        font=("Segoe UI", 13, "bold"))
        style.configure("TButton", padding=(12, 7))
        style.configure("Accent.TButton", background=BLUE, foreground="white")
        style.map("Accent.TButton",
                  background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])
        style.configure("TEntry", padding=6, fieldbackground=CARD)
        style.configure("TCombobox", padding=5, fieldbackground=CARD)
        style.configure("Treeview", rowheight=29, background=CARD,
                        fieldbackground=CARD, borderwidth=0)
        style.configure("Treeview.Heading", background="#e2e8f0",
                        font=("Segoe UI", 10, "bold"), padding=7)
        style.map("Treeview", background=[("selected", "#dbeafe")],
                  foreground=[("selected", INK)])
        style.configure("TNotebook.Tab", padding=(18, 9))

    def _card(self, parent, title):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=14)
        frame.pack(fill="x", pady=(0, 10))
        ttk.Label(frame, text=title, style="Title.TLabel").pack(
            anchor="w", pady=(0, 10))
        return frame

    def _build(self):
        header = tk.Frame(self, bg=INK, padx=24, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="CloudMarks", bg=INK, fg="white",
                 font=("Segoe UI", 21, "bold")).pack(side="left")
        tk.Label(header, text="Виртуальные точки  /  Деформационный мониторинг",
                 bg=INK, fg="#b8c7dc").pack(side="left", padx=24)

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, width=355)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        left.pack_propagate(False)

        self.inputs = ttk.Frame(left)
        self.inputs.pack(fill="x")

        files = self._card(self.inputs, "Исходные данные")
        self.mode = tk.StringVar(value="Один скан")
        mode = ttk.Combobox(
            files, textvariable=self.mode, state="readonly",
            values=["Один скан", "Две эпохи", "Фрагмент"])
        mode.pack(fill="x", pady=(0, 10))
        mode.bind("<<ComboboxSelected>>", self._mode_changed)

        self.paths = []
        self.file_rows = []
        for label, types in [
            ("Скан / эпоха 1", SCAN_TYPES),
            ("Эпоха 2", SCAN_TYPES),
            ("Файл базовых точек", REF_TYPES),
        ]:
            row = ttk.Frame(files, style="Card.TFrame")
            caption = ttk.Label(row, text=label, style="Hint.TLabel")
            caption.pack(anchor="w")
            line = ttk.Frame(row, style="Card.TFrame")
            line.pack(fill="x", pady=(4, 8))
            var = tk.StringVar()
            ttk.Entry(line, textvariable=var, width=24).pack(
                side="left", fill="x", expand=True)
            ttk.Button(
                line, text="…", width=3,
                command=lambda v=var, t=types: self._choose(v, t),
            ).pack(side="left", padx=(5, 0))
            self.paths.append(var)
            self.file_rows.append(row)

        options = self._card(self.inputs, "Параметры поиска")
        self.param_vars = {}
        self.param_rows = {}
        for key, label, default, _ in PARAMS:
            row = ttk.Frame(options, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
            var = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=var, width=8).pack(side="right")
            self.param_vars[key] = var
            self.param_rows[key] = row

        ttk.Button(options, text="Сбросить параметры",
                   command=self._reset).pack(fill="x", pady=(10, 0))

        self.pair_options = self._card(self.inputs, "Сравнение эпох")
        self.alpha = tk.StringVar(value="0.05")
        self.max_pair = tk.StringVar(value="0.03")
        for label, var in [
            ("Уровень значимости α", self.alpha),
            ("Предел расхождения пары, м", self.max_pair),
        ]:
            row = ttk.Frame(self.pair_options, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
            ttk.Entry(row, textvariable=var, width=8).pack(side="right")

        self.note = ttk.Label(left, wraplength=335, foreground=MUTED)
        self.note.pack(fill="x", pady=(0, 10))
        self.run_button = ttk.Button(
            left, text="Начать расчёт", style="Accent.TButton",
            command=self._start)
        self.run_button.pack(fill="x")

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self.summary = tk.StringVar(value="Загрузите данные и запустите расчёт")
        ttk.Label(right, textvariable=self.summary,
                  font=("Segoe UI", 13, "bold")).grid(
                      row=0, column=0, sticky="w", pady=(2, 14))

        tools = ttk.Frame(right)
        tools.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.search = tk.StringVar()
        search = ttk.Entry(tools, textvariable=self.search, width=22)
        search.pack(side="left")
        self.search.trace_add("write", lambda *_: self._filter())

        self.filter_status = tk.StringVar(value="Все статусы")
        statuses = ttk.Combobox(
            tools, textvariable=self.filter_status, state="readonly", width=18,
            values=["Все статусы", "SUCCESS", "UNRELIABLE", "FAILED"])
        statuses.pack(side="left", padx=7)
        statuses.bind("<<ComboboxSelected>>", lambda _: self._filter())

        ttk.Button(tools, text="Экспорт…", command=self._export).pack(side="right")
        self.export_filtered = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tools, text="Только выборка", variable=self.export_filtered,
        ).pack(side="right", padx=8)

        tabs = ttk.Notebook(right)
        tabs.grid(row=2, column=0, sticky="nsew")
        table_tab, plot_tab = ttk.Frame(tabs), ttk.Frame(tabs)
        tabs.add(table_tab, text="Таблица и диагностика")
        tabs.add(plot_tab, text="График")

        pane = ttk.Panedwindow(table_tab, orient="vertical")
        pane.pack(fill="both", expand=True)
        table_frame = ttk.Frame(pane)
        pane.add(table_frame, weight=4)

        self.tree = ttk.Treeview(table_frame, show="headings")
        sy = ttk.Scrollbar(table_frame, command=self.tree.yview)
        sx = ttk.Scrollbar(table_frame, orient="horizontal",
                           command=self.tree.xview)
        self.tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._details)
        self.tree.tag_configure("SUCCESS", foreground="#15803d")
        self.tree.tag_configure("UNRELIABLE", foreground="#a16207")
        self.tree.tag_configure("FAILED", foreground="#b91c1c")

        details_frame = ttk.Frame(pane)
        pane.add(details_frame, weight=1)
        self.details = tk.Text(
            details_frame, height=9, wrap="word", bg=CARD, fg=INK,
            relief="flat", padx=14, pady=12, font=("Consolas", 10))
        ds = ttk.Scrollbar(details_frame, command=self.details.yview)
        self.details.configure(yscrollcommand=ds.set)
        ds.pack(side="right", fill="y")
        self.details.pack(fill="both", expand=True)
        self._set_details("Выберите строку для просмотра полной диагностики.")

        self.figure = Figure(figsize=(8, 5), facecolor=CARD)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        footer = ttk.Frame(self, padding=(18, 7))
        footer.pack(fill="x")
        self.status = tk.StringVar(value="Готово")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=220)
        self.progress.pack(side="right")
        self._plot()

    def _choose(self, var, types):
        path = filedialog.askopenfilename(parent=self, filetypes=types)
        if path:
            var.set(path)

    def _reset(self):
        for key, _, default, _ in PARAMS:
            self.param_vars[key].set(str(default))
        self.alpha.set("0.05")
        self.max_pair.set("0.03")

    def _mode_changed(self, _=None):
        mode = self.mode.get()
        for row in self.file_rows:
            row.pack_forget()
        for i in ([0, 1, 2] if mode == "Две эпохи"
                  else [0, 2] if mode == "Один скан" else [0]):
            self.file_rows[i].pack(fill="x")

        for key, row in self.param_rows.items():
            row.pack_forget()
        for key, row in self.param_rows.items():
            if mode != "Фрагмент" or key in {
                "normal_k", "min_points_per_plane",
                "cluster_eps", "cluster_min_samples",
            }:
                row.pack(fill="x", pady=3, before=row.master.winfo_children()[-1])

        self.pair_options.pack_forget()
        if mode == "Две эпохи":
            self.pair_options.pack(fill="x", pady=(0, 10))
        notes = {
            "Один скан": "Базовые точки: name X Y Z [radius]. "
                        "Радиус из файла имеет приоритет над общим.",
            "Две эпохи": "Сканы должны быть заранее совмещены в общей "
                        "системе координат. Регистрация не выполняется.",
            "Фрагмент": "Готовый локальный фрагмент с тремя плоскостями. "
                        "Базовые точки не требуются.",
        }
        self.note.configure(text=notes[mode])

    def _start(self):
        if self.busy:
            return
        try:
            mode = self.mode.get()
            paths = [v.get().strip() for v in self.paths]
            needed = [0, 1, 2] if mode == "Две эпохи" else (
                [0, 2] if mode == "Один скан" else [0])
            if any(not Path(paths[i]).is_file() for i in needed):
                raise ValueError("Выберите существующие входные файлы.")

            params = {
                key: cast(self.param_vars[key].get())
                for key, _, _, cast in PARAMS
            }
            alpha = float(self.alpha.get()) if mode == "Две эпохи" else 0.05
            max_pair = float(self.max_pair.get()) if mode == "Две эпохи" else None
        except Exception as error:
            messagebox.showerror("Исходные данные", str(error), parent=self)
            return

        self.busy = True
        self.run_button.state(["disabled"])
        self.progress.start(12)
        self.status.set("Загрузка и обработка…")
        threading.Thread(
            target=self._worker,
            args=(mode, paths, params, alpha, max_pair),
            daemon=True,
        ).start()

    def _worker(self, *args):
        try:
            self.events.put(("result", calculate(*args)))
        except Exception as error:
            self.events.put(("error", f"{type(error).__name__}: {error}"))

    def _poll(self):
        try:
            kind, payload = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.progress.stop()
            self.run_button.state(["!disabled"])
            if kind == "error":
                self.status.set("Расчёт завершился ошибкой")
                messagebox.showerror("Ошибка расчёта", payload, parent=self)
            else:
                self.result = payload
                self.search.set("")
                self.filter_status.set("Все статусы")
                self._filter()
                self._set_details("Выберите строку для просмотра диагностики.")
                self.status.set("Расчёт завершён")
        self.after(100, self._poll)

    def _status_column(self):
        columns = self.result.table.columns
        return next(
            (c for c in ("processing_status", "status") if c in columns), None)

    def _filter(self):
        df = self.result.table
        if df.empty:
            self.view = df
        else:
            mask = pd.Series(True, index=df.index)
            if self.search.get() and "name" in df:
                mask &= df["name"].astype(str).str.contains(
                    self.search.get(), case=False, regex=False)
            col = self._status_column()
            selected = self.filter_status.get()
            if col and selected != "Все статусы":
                mask &= df[col].eq(selected)
            self.view = df.loc[mask]
        self._render()
        self._plot()

    def _render(self):
        self.tree.delete(*self.tree.get_children())
        columns = list(self.view.columns)
        self.tree["columns"] = columns
        col = self._status_column()

        for c in columns:
            self.tree.heading(c, text=c, command=lambda c=c: self._sort(c))
            self.tree.column(c, width=130, minwidth=85, stretch=False)

        for i, (_, row) in enumerate(self.view.iterrows()):
            status = str(row[col]) if col else ""
            self.tree.insert(
                "", "end", iid=str(i),
                values=[text(v) for v in row], tags=(status,))

        df = self.result.table
        accepted = 0
        if col and not df.empty:
            accepted = df[col].isin(["SUCCESS", "GOOD"]).sum()
        self.summary.set(
            f"Всего: {len(df)}    Принято: {accepted}    "
            f"В выборке: {len(self.view)}")

    def _sort(self, column):
        reverse = self.sort_reverse.get(column, False)
        self.view = self.view.sort_values(
            column, ascending=not reverse, na_position="last", kind="stable")
        self.sort_reverse[column] = not reverse
        self._render()

    def _set_details(self, value):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", value)
        self.details.configure(state="disabled")

    def _details(self, _=None):
        selection = self.tree.selection()
        if not selection:
            return
        row = self.view.iloc[int(selection[0])]
        lines = [f"{key}: {text(value)}" for key, value in row.items()]
        points = self.result.objects.get(row.get("name"), [])
        for i, point in enumerate(points, 1):
            if point is None:
                continue
            title = f"Эпоха {i}" if len(points) > 1 else "Виртуальная точка"
            lines.append(f"\n{title}")
            if point.cov_xyz is not None:
                lines.append("Ковариационная матрица, м²:\n" +
                             np.array2string(point.cov_xyz, precision=6))
            if point.ellipsoid is not None:
                ellipsoid = point.ellipsoid
                lines.append(
                    f"Эллипсоид: P={ellipsoid['confidence']:.0%}; "
                    f"полуоси, м: {ellipsoid['semi_axes']}")
        self._set_details("\n".join(lines))

    def _plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        df = self.view

        if "displacement_mm" in df and df["displacement_mm"].notna().any():
            data = df.dropna(subset=["displacement_mm"]).sort_values(
                "displacement_mm", ascending=False).head(30)
            x = np.arange(len(data))
            errors = (
                data["sigma_displacement_mm"].fillna(0).to_numpy()
                if "sigma_displacement_mm" in data else None
            )
            ax.bar(x, data["displacement_mm"], color=BLUE, width=0.65)
            if errors is not None:
                ax.errorbar(x, data["displacement_mm"], yerr=errors,
                            fmt="none", ecolor=INK, capsize=3)
            ax.set_xticks(x, data["name"], rotation=60, ha="right", fontsize=8)
            ax.set_ylabel("Смещение, мм")
            ax.set_title("До 30 наибольших смещений · погрешности ±1σ")
            ax.grid(axis="y", alpha=0.2)
        elif {"x", "y"}.issubset(df.columns) and df[["x", "y"]].notna().all(axis=1).any():
            data = df.dropna(subset=["x", "y"])
            ax.scatter(data["x"], data["y"], s=38, color=BLUE)
            if len(data) <= 30:
                for _, row in data.iterrows():
                    ax.annotate(str(row["name"]), (row["x"], row["y"]),
                                xytext=(5, 5), textcoords="offset points",
                                fontsize=8)
            ax.set(xlabel="X, м", ylabel="Y, м", title="План виртуальных точек")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
        else:
            ax.text(0.5, 0.5, "Нет данных для построения",
                    ha="center", va="center", color=MUTED,
                    transform=ax.transAxes)
            ax.set_axis_off()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _export(self):
        df = self.view if self.export_filtered.get() else self.result.table
        if df.empty:
            messagebox.showinfo("Экспорт", "Нет данных для экспорта.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".xlsx",
            initialfile="cloudmarks_results.xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV UTF-8", "*.csv")],
        )
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".xlsx":
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
            self.status.set(f"Сохранено: {path}")
        except Exception as error:
            messagebox.showerror("Ошибка экспорта", str(error), parent=self)