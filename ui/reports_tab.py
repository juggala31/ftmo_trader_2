from __future__ import annotations
import pathlib, datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import sqlite3

# Optional chart support
HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SUM_DIR = _ROOT / "ai" / "summaries"
DB_PATH = _ROOT / "db" / "trades.sqlite"

class ReportsTab(ttk.Frame):
    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.current_date = dt.date.today()
        self._build_ui()
        self._load_date(self.current_date)

    def _build_ui(self):
        # Controls
        bar = ttk.Frame(self); bar.pack(fill="x", padx=8, pady=8)
        ttk.Label(bar, text="Date (YYYY-MM-DD):").pack(side="left")
        self.date_var = tk.StringVar(value=str(self.current_date))
        ttk.Entry(bar, width=12, textvariable=self.date_var).pack(side="left", padx=(6,10))
        ttk.Button(bar, text="◀ Prev", command=self._prev_day).pack(side="left")
        ttk.Button(bar, text="Today", command=self._today).pack(side="left", padx=(6,0))
        ttk.Button(bar, text="Next ▶", command=self._next_day).pack(side="left", padx=(6,10))
        ttk.Button(bar, text="Load", command=self._load_from_entry).pack(side="left")
        ttk.Button(bar, text="Open CSV…", command=self._open_csv).pack(side="right")

        # Period controls (SQLite)
        pbar = ttk.LabelFrame(self, text="Period Report (from SQLite)"); pbar.pack(fill="x", padx=8, pady=(0,8))
        ttk.Label(pbar, text="Start:").pack(side="left")
        self.p_start = tk.StringVar(value=str(self.current_date.replace(day=1)))
        ttk.Entry(pbar, width=12, textvariable=self.p_start).pack(side="left", padx=(4,10))
        ttk.Label(pbar, text="End:").pack(side="left")
        self.p_end = tk.StringVar(value=str(self.current_date))
        ttk.Entry(pbar, width=12, textvariable=self.p_end).pack(side="left", padx=(4,10))
        ttk.Button(pbar, text="Load Period", command=self._load_period).pack(side="left", padx=(6,0))

        # Summary labels
        self.sum_lbl = ttk.Label(self, text="—", font=("Segoe UI", 10))
        self.sum_lbl.pack(fill="x", padx=10)

        # Table
        cols = ("ticket","symbol","side","volume","entry","exit","pnl_ccy","t_open","t_close")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=16)
        for c, w in zip(cols, (90,100,60,70,90,90,90,110,110)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=(6,6))

        # Chart (optional)
        self.chart_frame = ttk.Frame(self); self.chart_frame.pack(fill="both", expand=False, padx=10, pady=(0,10))
        if not HAVE_MPL:
            ttk.Label(self.chart_frame, text="(Install matplotlib for PnL charts:  pip install matplotlib)").pack(side="left")

    # --- Daily CSV view ---
    def _prev_day(self): self._shift_days(-1)
    def _next_day(self): self._shift_days(1)
    def _today(self): self._load_date(dt.date.today())
    def _shift_days(self, n: int):
        try:
            d = dt.date.fromisoformat(self.date_var.get())
        except Exception:
            d = self.current_date
        self._load_date(d + dt.timedelta(days=n))
    def _load_from_entry(self):
        try:
            d = dt.date.fromisoformat(self.date_var.get())
        except Exception:
            messagebox.showwarning("Bad date", "Use YYYY-MM-DD"); return
        self._load_date(d)
    def _open_csv(self):
        p = filedialog.askopenfilename(initialdir=str(SUM_DIR), filetypes=[("CSV files","*.csv")])
        if not p: return
        try:
            df = pd.read_csv(p); self._apply_df(df, pathlib.Path(p).name)
        except Exception as e:
            messagebox.showerror("Open failed", str(e))
    def _load_date(self, d: dt.date):
        self.current_date = d; self.date_var.set(str(d))
        p = SUM_DIR / f"trades_{d.isoformat()}.csv"
        if not p.exists():
            self._apply_df(pd.DataFrame(columns=["ticket","symbol","side","volume","entry","exit","pnl_ccy","t_open","t_close"]), f"(missing) {p.name}"); return
        try:
            df = pd.read_csv(p)
        except Exception as e:
            messagebox.showerror("Load failed", str(e)); return
        self._apply_df(df, p.name)

    # --- Period (SQLite) ---
    def _load_period(self):
        if not DB_PATH.exists():
            messagebox.showwarning("No DB", f"SQLite not found: {DB_PATH}"); return
        try:
            s = dt.date.fromisoformat(self.p_start.get())
            e = dt.date.fromisoformat(self.p_end.get())
        except Exception:
            messagebox.showwarning("Bad dates", "Use YYYY-MM-DD for start/end"); return
        ts = dt.datetime(s.year,s.month,s.day).timestamp()
        te = dt.datetime(e.year,e.month,e.day,23,59,59).timestamp()
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM trades WHERE t_close BETWEEN ? AND ? ORDER BY t_close ASC", con, params=(ts, te))
        con.close()
        if df is None or len(df)==0:
            messagebox.showinfo("Period", "No rows in the selected range."); return
        title = f"SQLite period {s}..{e}"
        self._apply_df(df, title)

    # --- Apply to UI ---
    def _apply_df(self, df: pd.DataFrame, title: str):
        self.tree.delete(*self.tree.get_children())
        for _, r in df.iterrows():
            self.tree.insert("", "end", values=(
                r.get("ticket",""), r.get("symbol",""), r.get("side",""),
                f"{float(r.get('volume',0.0)):.2f}",
                f"{float(r.get('entry',0.0)):.2f}",
                f"{float(r.get('exit',0.0)):.2f}",
                f"{float(r.get('pnl_ccy',0.0)):+.2f}",
                f"{float(r.get('t_open',0.0)):.0f}",
                f"{float(r.get('t_close',0.0)):.0f}",
            ))
        txt = f"{title} — rows={len(df)}"
        if len(df):
            g = df.groupby("symbol")["pnl_ccy"].agg(["count","sum","mean"]).reset_index()
            total = float(df["pnl_ccy"].sum())
            wins = int((df["pnl_ccy"] >= 0).sum())
            winrate = (wins/len(df))*100.0
            txt += f" | total={total:+.2f}  winrate={winrate:.1f}%"
        self.sum_lbl.config(text=txt)

        for w in self.chart_frame.winfo_children(): w.destroy()
        if HAVE_MPL and len(df):
            g = df.groupby("symbol")["pnl_ccy"].sum().reset_index()
            fig = Figure(figsize=(6.0, 2.2), dpi=100); ax = fig.add_subplot(111)
            ax.bar(g["symbol"], g["pnl_ccy"]); ax.set_title("PnL by Symbol"); ax.set_ylabel("PnL (ccy)"); ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout(); canvas = FigureCanvasTkAgg(fig, master=self.chart_frame); canvas.draw(); canvas.get_tk_widget().pack(fill="x", expand=False)