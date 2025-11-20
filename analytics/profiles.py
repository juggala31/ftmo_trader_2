from __future__ import annotations
import pathlib, sys
from typing import Dict, Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except Exception:
    yaml = None

def _load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    if yaml:
        try: return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception: return {}
    return {}

def apply_profile_overlays(profile_name: str, config_dir: str = str(_ROOT / "config")) -> Dict[str,Any]:
    """
    Returns a dict of overlay configs: {"time_windows": {...}, "daily_limits": {...}, ...}
    Caller is responsible for writing files or merging at runtime.
    """
    p = pathlib.Path(config_dir) / "profiles" / f"{profile_name.lower()}.yml"
    data = _load_yaml(p)
    return data.get("overrides") or {}