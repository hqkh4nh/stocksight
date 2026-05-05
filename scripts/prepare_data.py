import argparse

from src.data import download_all

def main():
    parser = argparse.ArgumentParser(description="Download stock data from yfinance")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    print("=" * 60)
    print("StockSight Data Preparation")
    print("=" * 60)

    results = download_all(force=args.force)

    print("\n" + "=" * 60)
    print("Summary:")
    for ticker, rows in results.items():
        status = "✓" if rows > 0 else "✗"
        print(f"  {status} {ticker:6s} {rows} rows")
    print("=" * 60)

    failed = [t for t, r in results.items() if r == 0]
    if failed:
        print(f"\nFailed: {failed}")
    else:
        print("\nAll tickers downloaded successfully.")

if __name__ == "__main__":
    main()

