"""Download stock + macro parquet caches for all configured tickers."""
import argparse

from src.config import CFG
from src.data_loader import load_macro, load_stock, MACRO_LABELS


def main():
    parser = argparse.ArgumentParser(description="Prefetch parquet caches for tickers + macro")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    print("=== Macro indices ===")
    for sym in CFG["macro_indices"]:
        try:
            df = load_macro(sym, refresh=args.force)
            print(f"  {MACRO_LABELS[sym]:4s} ({sym}) {len(df)} rows")
        except Exception as e:
            print(f"  {MACRO_LABELS.get(sym, sym):4s} ({sym}) FAILED: {e}")

    print("\n=== Stocks ===")
    tickers = CFG["tickers"]
    results = {}
    for i, tkr in enumerate(tickers, 1):
        try:
            df = load_stock(tkr, refresh=args.force)
            print(f"[{i}/{len(tickers)}] {tkr:6s} {len(df)} rows")
            results[tkr] = len(df)
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {tkr:6s} FAILED: {e}")
            results[tkr] = 0

    failed = [t for t, r in results.items() if r == 0]
    if failed:
        print(f"\nfailed: {failed}")


if __name__ == "__main__":
    main()
