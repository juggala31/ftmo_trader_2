"""
Entry point wrapper for the MT5 Position Guardian.

Usage:
    python -m ai.guardian_entry
"""

from .position_guardian_mt5 import run_guardian_loop


def main():
    run_guardian_loop()


if __name__ == "__main__":
    main()