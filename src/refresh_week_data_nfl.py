# src/refresh_week_data_nfl.py
"""
NFL refresh alias.

Purpose:
  Allow `python -m src.refresh_week_data_nfl` while keeping legacy
  `python -m src.refresh_week_data` working unchanged.

Spec:
  /context/global_week_and_provider_decoupling.md (read-only logging here)

Notes:
  This prints a read-only CurrentWeek(NFL) line, then defers to the
  existing NFL refresh module (src.refresh_week_data) with the same argv.
"""
from src.common.current_week_service import get_current_week

def _log_current_week_readonly() -> None:
    season, week, ts = get_current_week("NFL")
    print(f"CurrentWeek(NFL)={season} W{week} computed_at={ts} (readonly)")

def main():
    _log_current_week_readonly()
    # Defer to the legacy NFL refresh module; reuse its CLI/args unchanged.
    import sys
    from src.refresh_week_data import main as legacy_main
    # Pass through the original argv after the -m invocation
    legacy_main()

if __name__ == "__main__":
    main()