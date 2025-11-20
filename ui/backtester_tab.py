from __future__ import annotations
import threading, subprocess, sys, pathlib, datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox

_ROOT = pathlib.Path(__file__).resolve().parents[1]
AI_DIR = _ROOT / "ai" / "datasets"

DEFAULT_BASES = "US30,US100,US500,BTC,OIL,GOLD"
DEFAULT_TFS   = ["M5","M15","M30","H1","H4"]

def _load_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

class BacktesterTab(ttk.Frame):
    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        # Controls area
        opts = ttk.LabelFrame(self, text="Backtester / AI Export")
        opts.pack(fill="x", padx=8, pady=8)

        # Symbols row
        row1 = ttk.Frame(opts); row1.pack(fill="x", padx=8, pady=6)
        ttk.Label(row1, text="Symbols (comma-separated):").pack(side="left")
        self.sym_entry = ttk.Entry(row1, width=80)
        self.sym_entry.insert(0, DEFAULT_BASES)
        self.sym_entry.pack(side="left", padx=(6,6))

        self.resolve_btn = ttk.Button(row1, text="Resolve (OANDA MT5)", command=self._resolve_symbols)
        self.resolve_btn.pack(side="left")

        # Timeframes
        row2 = ttk.Frame(opts); row2.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2, text="Timeframes:").pack(side="left")
        self.tf_vars = {}
        for tf in DEFAULT_TFS:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row2, text=tf, variable=var).pack(side="left", padx=(4,0))
            self.tf_vars[tf] = var

        # Range
        row3 = ttk.Frame(opts); row3.pack(fill="x", padx=8, pady=6)
        ttk.Label(row3, text="Years:").pack(side="left")
        self.years_var = tk.StringVar(value="5")
        ttk.Entry(row3, width=6, textvariable=self.years_var).pack(side="left", padx=(4,12))
        ttk.Label(row3, text="Start (YYYY-MM-DD, optional):").pack(side="left")
        self.start_var = tk.StringVar(value="")
        ttk.Entry(row3, width=12, textvariable=self.start_var).pack(side="left", padx=(4,12))
        ttk.Label(row3, text="End (YYYY-MM-DD, optional):").pack(side="left")
        self.end_var = tk.StringVar(value="")
        ttk.Entry(row3, width=12, textvariable=self.end_var).pack(side="left", padx=(4,12))

        # Output
        row4 = ttk.Frame(opts); row4.pack(fill="x", padx=8, pady=6)
        ttk.Label(row4, text="Output dir:").pack(side="left")
        self.out_entry = ttk.Entry(row4, width=60)
        self.out_entry.insert(0, str(AI_DIR))
        self.out_entry.pack(side="left", padx=(6,6))
        self.csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="CSV", variable=self.csv_var).pack(side="left", padx=(6,0))
        self.parquet_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="Parquet", variable=self.parquet_var).pack(side="left", padx=(6,0))

        # Run
        row5 = ttk.Frame(opts); row5.pack(fill="x", padx=8, pady=6)
        self.run_btn = ttk.Button(row5, text="Run Export", command=self._run_export)
        self.run_btn.pack(side="left")
        self.pb = ttk.Progressbar(row5, mode="indeterminate", length=220)
        self.pb.pack(side="left", padx=(12,0))

        # Log
        logf = ttk.LabelFrame(self, text="Log")
        logf.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.log = tk.Text(logf, height=14, wrap="word")
        self.log.pack(fill="both", expand=True)

    def _append_log(self, msg: str):
        try:
            self.log.insert("end", msg + "\n")
            self.log.see("end")
        except Exception:
            pass

    def _resolve_symbols(self):
        """Call tools/resolve_symbols.py to map US30,US100,US500,BTC,OIL,GOLD to broker symbols."""
        self._append_log("[RESOLVE] Resolving via tools/resolve_symbols.py …")
        py = sys.executable
        script = str(_ROOT / "tools" / "resolve_symbols.py")
        want = self.sym_entry.get().strip() or DEFAULT_BASES
        try:
            out = subprocess.check_output([py, script, "--want", want], stderr=subprocess.STDOUT, cwd=str(_ROOT), text=True)
            self._append_log(out.rstrip())
            # find SYMBOLS= line
            for line in out.splitlines():
                if line.startswith("SYMBOLS="):
                    csv = line.split("=",1)[1].strip()
                    self.sym_entry.delete(0, "end")
                    self.sym_entry.insert(0, csv)
                    self._append_log(f"[RESOLVE] Selected: {csv}")
                    break
        except subprocess.CalledProcessError as e:
            self._append_log(e.output or str(e))
            messagebox.showerror("Resolve failed", "Could not resolve symbols. Open them in MT5 Market Watch and try again.")

    def _run_export(self):
        syms = [s.strip() for s in (self.sym_entry.get() or "").split(",") if s.strip()]
        if not syms:
            messagebox.showwarning("No symbols", "Enter comma-separated symbols first.")
            return
        tfs = [tf for tf,v in self.tf_vars.items() if v.get()]
        if not tfs:
            messagebox.showwarning("No timeframes", "Select at least one timeframe.")
            return
        years = self.years_var.get().strip()
        start = self.start_var.get().strip()
        end   = self.end_var.get().strip()
        outd  = self.out_entry.get().strip() or str(AI_DIR)
        csv   = self.csv_var.get()
        parq  = self.parquet_var.get()

        # background runner
        def worker():
            self.pb.start(12)
            self.run_btn.configure(state="disabled")
            try:
                self._append_log(f"[RUN] symbols={syms}  tfs={tfs}  years={years}  start={start or '(auto)'}  end={end or '(today)'}")
                # import backtester as module and call run_export()
                sys.path.insert(0, str(_ROOT))
                from backtests.run_bt import run_export
                res = run_export(symbols=syms, timeframes=tfs, years=int(years or 5),
                                 start=start or None, end=end or None,
                                 out_dir=outd, csv=bool(csv), parquet=bool(parq),
                                 log_cb=lambda m: self._append_log(m))
                self._append_log(f"[DONE] {res}")
            except Exception as e:
                self._append_log(f"[ERROR] {e}")
                messagebox.showerror("Backtester error", str(e))
            finally:
                self.pb.stop()
                self.run_btn.configure(state="normal")

        threading.Thread(target=worker, daemon=True).start()