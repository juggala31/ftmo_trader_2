import importlib.util, sys, pathlib

root = pathlib.Path(r"C:\ftmo_trader_2")
p = root / "run_live.py"

spec = importlib.util.spec_from_file_location("run_live", str(p))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("[DIAG] loaded:", p)
print("[DIAG] has main:", hasattr(mod, "main"))
rc = None
if hasattr(mod, "main"):
    try:
        rc = mod.main()
        print("[DIAG] main() returned:", rc)
    except SystemExit as e:
        print("[DIAG] SystemExit code:", getattr(e, "code", None))
    except Exception as e:
        print("[DIAG] main() raised:", repr(e))
else:
    print("[DIAG] no main(); nothing to run")
