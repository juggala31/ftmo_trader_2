from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_LIVE = ROOT / "run_live.py"


def main() -> None:
    print(f"[patch_trade_mode] Root: {ROOT}")
    if not RUN_LIVE.exists():
        raise SystemExit(f"run_live.py not found at {RUN_LIVE}")

    text = RUN_LIVE.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # 1) Ensure top import has json
    # ------------------------------------------------------------------
    first_newline = text.find("\n")
    if first_newline == -1:
        raise SystemExit("Unable to find first line in run_live.py")

    first_line = text[:first_newline]
    if "import os, threading, time, inspect, json" not in first_line:
        text = text.replace(
            "import os, threading, time, inspect",
            "import os, threading, time, inspect, json",
            1,
        )
        print("[patch_trade_mode] Added json to top import line.")
    else:
        print("[patch_trade_mode] json already in top import line.")

    # ------------------------------------------------------------------
    # 2) Insert trade-mode helper after `from pathlib import Path`
    # ------------------------------------------------------------------
    idx_pl = text.find("from pathlib import Path")
    if idx_pl == -1:
        raise SystemExit("Could not find 'from pathlib import Path' in run_live.py")

    line_end = text.find("\n", idx_pl)
    if line_end == -1:
        raise SystemExit("Could not find end of 'from pathlib import Path' line")

    insert_pos = line_end + 1  # insert after that line

    trade_helper = """

# ------------- trade-mode gate (ai_profile.json) -------------
_ROOT = Path(__file__).resolve().parent
_AI_PROFILE_PATH = _ROOT / "ai" / "ai_profile.json"
_TRADE_PROFILE_CACHE: Dict[str, Dict[str, Any]] = {}
_TRADE_PROFILE_MTIME: float = 0.0

def _load_trade_profile() -> None:
    \"\"\"Lazy-reload ai_profile.json when it changes on disk.\"\"\"
    global _TRADE_PROFILE_CACHE, _TRADE_PROFILE_MTIME
    try:
        st = _AI_PROFILE_PATH.stat()
    except FileNotFoundError:
        return

    mtime = st.st_mtime
    if _TRADE_PROFILE_CACHE and _TRADE_PROFILE_MTIME == mtime:
        return

    try:
        with _AI_PROFILE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f\"[trade_mode] failed to load {_AI_PROFILE_PATH}: {e}\", flush=True)
        return

    cache: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for sym, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            enabled = bool(cfg.get("enabled", True))
            mode = str(cfg.get("trade_mode", "live")).strip().lower()
            if not mode:
                mode = "live"
            if mode not in ("live", "paper", "off"):
                mode = "live"
            cache[str(sym)] = {"enabled": enabled, "trade_mode": mode}

    _TRADE_PROFILE_CACHE = cache
    _TRADE_PROFILE_MTIME = mtime
    print(f\"[trade_mode] loaded {len(cache)} symbols from {_AI_PROFILE_PATH}\", flush=True)

def _lookup_trade_cfg(canon_symbol: str) -> Dict[str, Any]:
    \"\"\"Return normalized trade cfg for canonical symbol.\"\"\"
    _load_trade_profile()
    for key in (canon_symbol, canon_symbol.upper()):
        cfg = _TRADE_PROFILE_CACHE.get(key)
        if cfg is not None:
            return cfg
    return {"enabled": True, "trade_mode": "live"}

"""

    text = text[:insert_pos] + trade_helper + text[insert_pos:]

    # ------------------------------------------------------------------
    # 3) Insert _trade_cfg_for_symbol inside TradingController
    #     (right after _override_or)
    # ------------------------------------------------------------------
    idx_class = text.find("class TradingController")
    if idx_class == -1:
        raise SystemExit("Could not find class TradingController in run_live.py")

    idx_ov = text.find("    def _override_or", idx_class)
    if idx_ov == -1:
        raise SystemExit("Could not find _override_or in TradingController")

    sub_from_ov = text[idx_ov:]
    next_def_rel = sub_from_ov.find("\n    def ", 10)
    if next_def_rel == -1:
        raise SystemExit("Could not find next 'def' after _override_or")

    insert_pos2 = idx_ov + next_def_rel

    trade_cfg_method = """
    def _trade_cfg_for_symbol(self, broker_symbol: str) -> Dict[str, Any]:
        \"\"\"Look up trade config for this broker symbol via canonical mapping.\"\"\"
        canon = self._broker_to_canon.get(broker_symbol, broker_symbol)
        return _lookup_trade_cfg(canon)

"""

    text = text[:insert_pos2] + trade_cfg_method + text[insert_pos2:]

    # ------------------------------------------------------------------
    # 4) Replace order_market with paper-aware version
    # ------------------------------------------------------------------
    idx_om = text.find("    def order_market", idx_class)
    if idx_om == -1:
        raise SystemExit("Could not find order_market in TradingController")

    sub_from_om = text[idx_om:]
    next_def_rel2 = sub_from_om.find("\n    def ", 10)
    if next_def_rel2 == -1:
        raise SystemExit("Could not find next 'def' after order_market")

    end_om = idx_om + next_def_rel2

    new_om_block = '''
    # ---- order path ----
    def order_market(self, symbol: str, side: str, volume: float, reason: str = "entry") -> Tuple[bool, Any]:
        # Global execute flag
        exec_global = getenv_bool("EXECUTE", False)

        # Profile-driven trade config (live / paper / off)
        trade_cfg = self._trade_cfg_for_symbol(symbol)
        mode = str(trade_cfg.get("trade_mode", "live")).strip().lower()
        enabled = bool(trade_cfg.get("enabled", True))
        if mode not in ("live", "paper", "off"):
            mode = "live"

        # Hard block if symbol is disabled/off
        if not enabled or mode == "off":
            comment = f"blocked by profile (enabled={enabled}, trade_mode={mode})"
            self.exec_logger.log_event(
                symbol=symbol,
                side=side,
                volume=volume,
                price="",
                retcode="OFF",
                comment=comment,
                ticket="",
                reason=reason,
                extra={
                    "trade_mode": mode,
                    "enabled": enabled,
                    "execute": False,
                },
            )
            return False, {
                "error": "symbol disabled by profile",
                "trade_mode": mode,
                "enabled": enabled,
            }

        # Paper-mode means: simulate only, even if EXECUTE=1
        is_paper = (mode == "paper")
        do_exec = bool(exec_global and not is_paper)

        price = None
        ticket = None

        # Guards
        ok_guard, why, rem = self._guard_check(symbol)
        if not ok_guard:
            self.exec_logger.log_event(
                symbol=symbol,
                side=side,
                volume=volume,
                price="",
                retcode=why,
                comment=f"blocked by {why.lower()} ({rem:0.1f}s)",
                ticket="",
                reason=reason,
                extra={
                    "trade_mode": mode,
                    "enabled": enabled,
                    "execute": do_exec,
                },
            )
            return False, {"error": f"blocked: {why}", "remaining_sec": rem}

        # Price snapshot
        try:
            if MT5:
                tick = MT5.symbol_info_tick(symbol)
                price = (tick.ask if side.upper() == "BUY" else tick.bid) if tick else None
        except Exception:
            price = None

        # Sizing (may override requested volume)
        eff_vol, size_diag = self._compute_volume(volume, side, price or 0.0, symbol)

        # Mark in-flight and log attempt
        self._guard_mark_after_attempt(symbol)
        self.exec_logger.log_event(
            symbol=symbol,
            side=side,
            volume=eff_vol,
            price=price or "",
            retcode="ATTEMPT",
            comment=f"attempt {reason}",
            ticket="",
            reason=reason,
            extra={
                "execute": do_exec,
                "execute_global": exec_global,
                "trade_mode": mode,
                "enabled": enabled,
                "sltp_mode": self.SLTP_MODE,
                "sizing": self.SIZING_MODE,
                "size_diag": size_diag,
            },
        )

        # SIM branch: EXECUTE=0 and/or trade_mode=paper
        if not do_exec:
            self._guard_mark_after_order(symbol)
            sim_reasons = []
            if not exec_global:
                sim_reasons.append("EXECUTE=0")
            if is_paper:
                sim_reasons.append("TRADE_MODE=paper")
            if not sim_reasons:
                sim_reasons.append("no_live_execution")
            comment = "simulate only (" + "+".join(sim_reasons) + ")"

            self.exec_logger.log_event(
                symbol=symbol,
                side=side,
                volume=eff_vol,
                price=price or "",
                retcode="SIM",
                comment=comment,
                ticket="",
                reason=reason,
                extra={
                    "trade_mode": mode,
                    "enabled": enabled,
                    "size_diag": size_diag,
                },
            )
            return True, {
                "simulated": True,
                "volume": eff_vol,
                "trade_mode": mode,
                "enabled": enabled,
            }

        # From here down: real execution (only if EXECUTE=1 and trade_mode=live)
        if not self.adapter:
            self.exec_logger.log_event(
                symbol=symbol,
                side=side,
                volume=eff_vol,
                price=price or "",
                retcode="NOADAPTER",
                comment="adapter missing",
                ticket="",
                reason=reason,
                extra={
                    "trade_mode": mode,
                    "enabled": enabled,
                    "size_diag": size_diag,
                },
            )
            return False, {"error": "adapter missing"}

        # Real execution
        try:
            res = self.adapter.order_market(symbol, side.upper(), eff_vol)
            retcode = res.get("retcode") if isinstance(res, dict) else None
            comment = res.get("comment") if isinstance(res, dict) else ""
            ticket = res.get("ticket") if isinstance(res, dict) else None
            ok = (
                (bool(res.get("ok", False)) if isinstance(res, dict) else bool(res))
                or _retcode_ok(retcode, comment)
            )

            self._guard_mark_after_order(symbol)
            self.exec_logger.log_event(
                symbol=symbol,
                side=side,
                volume=eff_vol,
                price=price or "",
                retcode=retcode if retcode is not None else ("OK" if ok else "FAIL"),
                comment=comment,
                ticket=ticket or "",
                reason=reason,
                extra={
                    "adapter": res,
                    "trade_mode": mode,
                    "enabled": enabled,
                    "size_diag": size_diag,
                },
            )

            if ok:
                self._apply_sltp(
                    ticket=(int(ticket) if ticket is not None else None),
                    symbol=symbol,
                    side=side,
                    entry_price=price,
                )
            return ok, {
                "retcode": retcode,
                "comment": comment,
                "ticket": ticket,
                "volume": eff_vol,
                "trade_mode": mode,
                "enabled": enabled,
            }
        except Exception as e:
            self.exec_logger.log_event(
                symbol=symbol,
                side=side,
                volume=eff_vol,
                price=price or "",
                retcode="EXC",
                comment=str(e),
                ticket="",
                reason=reason,
                extra={
                    "trace": traceback.format_exc(),
                    "trade_mode": mode,
                    "enabled": enabled,
                    "size_diag": size_diag,
                },
            )
            return False, {
                "error": str(e),
                "trade_mode": mode,
                "enabled": enabled,
            }

'''

    text = text[:idx_om] + new_om_block + text[end_om:]

    RUN_LIVE.write_text(text, encoding="utf-8")
    print("[patch_trade_mode] Patched run_live.py OK")


if __name__ == "__main__":
    main()