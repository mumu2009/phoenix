#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import traceback
import webbrowser
from pathlib import Path
from threading import Thread

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import stimulation_multiphysics as sm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "build" / "stimulation_workbench"


class WorkbenchApp:
	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("Phoenix Hardware Simulation Workbench")
		self.root.geometry("1040x760")
		self.root.minsize(920, 640)
		self.presets = sm.build_builtin_presets()
		self.last_dashboard: Path | None = None
		self.last_output_dir: Path | None = None

		self.mode_var = tk.StringVar(value="preset")
		self.preset_var = tk.StringVar(value=next(iter(self.presets)))
		self.preset_desc_var = tk.StringVar(value=self.presets[self.preset_var.get()]["description"])
		self.scenario_var = tk.StringVar()
		self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
		self.worker_var = tk.StringVar(value="4")
		self.open_dashboard_var = tk.BooleanVar(value=True)
		self.last_cli_command = ""

		self._build_style()
		self._build_ui()

	def _build_style(self) -> None:
		style = ttk.Style()
		for theme in ("vista", "clam", "default"):
			try:
				style.theme_use(theme)
				break
			except tk.TclError:
				continue
		style.configure("Card.TFrame", background="#fff8f0")
		style.configure("Title.TLabel", font=("Aptos", 16, "bold"))
		style.configure("Muted.TLabel", foreground="#506070")

	def _build_ui(self) -> None:
		root_frame = ttk.Frame(self.root, padding=18)
		root_frame.pack(fill=tk.BOTH, expand=True)

		header = ttk.Frame(root_frame)
		header.pack(fill=tk.X)
		ttk.Label(header, text="硬件仿真工作台", style="Title.TLabel").pack(anchor=tk.W)
		ttk.Label(
			header,
			text="Windows 11 本地 GUI，用于多物理域仿真、整板分块分析和 dashboard 浏览。",
			style="Muted.TLabel",
		).pack(anchor=tk.W, pady=(4, 0))

		form_card = ttk.Frame(root_frame, padding=18, style="Card.TFrame")
		form_card.pack(fill=tk.X, pady=(14, 12))

		mode_frame = ttk.Frame(form_card)
		mode_frame.pack(fill=tk.X, pady=(0, 12))
		ttk.Radiobutton(mode_frame, text="使用内置预设", variable=self.mode_var, value="preset", command=self._refresh_mode).pack(side=tk.LEFT)
		ttk.Radiobutton(mode_frame, text="使用外部场景 JSON", variable=self.mode_var, value="file", command=self._refresh_mode).pack(side=tk.LEFT, padx=(16, 0))

		preset_row = ttk.Frame(form_card)
		preset_row.pack(fill=tk.X, pady=4)
		ttk.Label(preset_row, text="预设场景", width=14).pack(side=tk.LEFT)
		self.preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var, values=list(self.presets.keys()), state="readonly")
		self.preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
		self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

		self.preset_desc = ttk.Label(form_card, textvariable=self.preset_desc_var, style="Muted.TLabel", wraplength=820)
		self.preset_desc.pack(fill=tk.X, pady=(2, 10))

		scenario_row = ttk.Frame(form_card)
		scenario_row.pack(fill=tk.X, pady=4)
		ttk.Label(scenario_row, text="场景 JSON", width=14).pack(side=tk.LEFT)
		self.scenario_entry = ttk.Entry(scenario_row, textvariable=self.scenario_var)
		self.scenario_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
		ttk.Button(scenario_row, text="浏览", command=self._browse_scenario).pack(side=tk.LEFT, padx=(10, 0))

		output_row = ttk.Frame(form_card)
		output_row.pack(fill=tk.X, pady=4)
		ttk.Label(output_row, text="输出目录", width=14).pack(side=tk.LEFT)
		self.output_entry = ttk.Entry(output_row, textvariable=self.output_var)
		self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
		ttk.Button(output_row, text="浏览", command=self._browse_output).pack(side=tk.LEFT, padx=(10, 0))

		worker_row = ttk.Frame(form_card)
		worker_row.pack(fill=tk.X, pady=8)
		ttk.Label(worker_row, text="并行 workers", width=14).pack(side=tk.LEFT)
		self.worker_spin = ttk.Spinbox(worker_row, from_=1, to=32, textvariable=self.worker_var, width=8)
		self.worker_spin.pack(side=tk.LEFT)
		ttk.Checkbutton(worker_row, text="完成后自动打开 dashboard", variable=self.open_dashboard_var).pack(side=tk.LEFT, padx=(18, 0))

		button_row = ttk.Frame(form_card)
		button_row.pack(fill=tk.X, pady=(8, 0))
		self.run_button = ttk.Button(button_row, text="运行仿真", command=self._run_clicked)
		self.run_button.pack(side=tk.LEFT)
		self.open_button = ttk.Button(button_row, text="打开 dashboard", command=self._open_dashboard, state=tk.DISABLED)
		self.open_button.pack(side=tk.LEFT, padx=(10, 0))
		self.output_button = ttk.Button(button_row, text="打开输出目录", command=self._open_output_dir, state=tk.DISABLED)
		self.output_button.pack(side=tk.LEFT, padx=(10, 0))
		self.copy_cli_button = ttk.Button(button_row, text="复制 CLI 命令", command=self._copy_cli_command)
		self.copy_cli_button.pack(side=tk.LEFT, padx=(10, 0))

		log_card = ttk.Frame(root_frame, padding=12, style="Card.TFrame")
		log_card.pack(fill=tk.BOTH, expand=True)
		ttk.Label(log_card, text="运行日志", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))
		self.log = scrolledtext.ScrolledText(log_card, wrap=tk.WORD, font=("Consolas", 10), background="#fffdf9")
		self.log.pack(fill=tk.BOTH, expand=True)
		self.log.configure(state=tk.DISABLED)

		self._refresh_mode()

	def _on_preset_changed(self, _event: object | None = None) -> None:
		self.preset_desc_var.set(self.presets[self.preset_var.get()]["description"])

	def _refresh_mode(self) -> None:
		mode = self.mode_var.get()
		preset_state = "readonly" if mode == "preset" else tk.DISABLED
		entry_state = tk.DISABLED if mode == "preset" else tk.NORMAL
		button_state = tk.DISABLED if mode == "preset" else tk.NORMAL
		self.preset_combo.configure(state=preset_state)
		self.scenario_entry.configure(state=entry_state)
		for child in self.scenario_entry.master.winfo_children():
			if isinstance(child, ttk.Button):
				child.configure(state=button_state)

	def _browse_scenario(self) -> None:
		path = filedialog.askopenfilename(
			title="选择仿真场景 JSON",
			filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
			initialdir=str(WORKSPACE_ROOT),
		)
		if path:
			self.scenario_var.set(path)

	def _browse_output(self) -> None:
		path = filedialog.askdirectory(title="选择输出目录", initialdir=self.output_var.get() or str(DEFAULT_OUTPUT_ROOT))
		if path:
			self.output_var.set(path)

	def _run_clicked(self) -> None:
		self.run_button.configure(state=tk.DISABLED)
		self.last_cli_command = self._build_cli_command()
		self._append_log("开始运行仿真...\n")
		self._append_log(f"CLI: {self.last_cli_command}\n\n")
		worker = Thread(target=self._run_worker, daemon=True)
		worker.start()

	def _build_cli_command(self) -> str:
		workers = max(int(self.worker_var.get() or "1"), 1)
		if self.mode_var.get() == "preset":
			selector = f'--preset "{self.preset_var.get()}"'
		else:
			selector = f'--scenario "{self.scenario_var.get()}"'
		return f'python tools/stimulation {selector} --out "{self.output_var.get()}" --workers {workers}'

	def _run_worker(self) -> None:
		try:
			report = self._execute_simulation()
			self.root.after(0, lambda: self._handle_success(report))
		except Exception as exc:  # pragma: no cover - GUI error path
			trace = traceback.format_exc()
			self.root.after(0, lambda: self._handle_failure(exc, trace))

	def _execute_simulation(self) -> dict[str, Any]:
		mode = self.mode_var.get()
		if mode == "preset":
			preset_name = self.preset_var.get()
			raw = json.loads(json.dumps(self.presets[preset_name]["scenario"], ensure_ascii=False))
			base_dir = WORKSPACE_ROOT
		else:
			scenario_path = Path(self.scenario_var.get()).resolve()
			if not scenario_path.exists():
				raise FileNotFoundError(f"场景文件不存在: {scenario_path}")
			with scenario_path.open("r", encoding="utf-8") as handle:
				raw = json.load(handle)
			base_dir = scenario_path.parent

		workers = max(int(self.worker_var.get() or "1"), 1)
		raw.setdefault("partition", {})
		raw["partition"]["workers"] = workers

		scenario = sm.load_scenario_dict(raw, base_dir)
		output_root = Path(self.output_var.get()).resolve() / sm.sanitize_path_part(scenario.name)
		output_root.mkdir(parents=True, exist_ok=True)
		report = sm.run_scenario(scenario, output_root)
		return report

	def _handle_success(self, report: dict[str, Any]) -> None:
		self.run_button.configure(state=tk.NORMAL)
		self.last_dashboard = Path(report["dashboard_path"])
		self.last_output_dir = self.last_dashboard.parent
		self.open_button.configure(state=tk.NORMAL)
		self.output_button.configure(state=tk.NORMAL)
		self._append_log(report["console_summary"] + "\n\n")
		if self.open_dashboard_var.get() and self.last_dashboard.exists():
			webbrowser.open(self.last_dashboard.as_uri())

	def _copy_cli_command(self) -> None:
		command = self.last_cli_command or self._build_cli_command()
		self.root.clipboard_clear()
		self.root.clipboard_append(command)
		self._append_log(f"已复制 CLI 命令: {command}\n")

	def _handle_failure(self, exc: Exception, trace: str) -> None:
		self.run_button.configure(state=tk.NORMAL)
		self._append_log(trace + "\n")
		messagebox.showerror("仿真失败", str(exc))

	def _append_log(self, text: str) -> None:
		self.log.configure(state=tk.NORMAL)
		self.log.insert(tk.END, text)
		self.log.see(tk.END)
		self.log.configure(state=tk.DISABLED)

	def _open_dashboard(self) -> None:
		if self.last_dashboard and self.last_dashboard.exists():
			webbrowser.open(self.last_dashboard.as_uri())

	def _open_output_dir(self) -> None:
		if self.last_output_dir and self.last_output_dir.exists():
			if hasattr(os, "startfile"):
				os.startfile(str(self.last_output_dir))
			else:
				webbrowser.open(self.last_output_dir.as_uri())


def main() -> int:
	root = tk.Tk()
	WorkbenchApp(root)
	root.mainloop()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())