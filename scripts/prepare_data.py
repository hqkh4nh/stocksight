import argparse

from src.data import download_all


def main():
    parser = argparse.ArgumentParser(description="Download stock data from yfinance")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    results = download_all(force=args.force)

    print("\nSummary:")
    for ticker, rows in results.items():
        print(f"  {ticker:6s} {rows} rows")

    failed = [t for t, r in results.items() if r == 0]
    if failed:
        print(f"\nfailed: {failed}")


if __name__ == "__main__":
    main()
