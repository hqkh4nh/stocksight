"""Per-ticker sector mapping and feature recipe.

The recipe drives which sector-aware correlation features each ticker gets:
- 'oil_correlated' tickers get Stock_Oil_Corr + Vol_x_Oil
- 'rate_correlated' tickers get Stock_Rate_Corr
"""

TICKER_SECTOR = {
    # Energy
    "BKR": "energy",
    "APA": "energy",
    # Industrials
    "NDSN": "industrial",
    "HON": "industrial",
    "CSX": "industrial",
    "JBHT": "industrial",
    "FAST": "industrial",
    "FELE": "industrial",
    # Consumer discretionary / staples
    "GT": "consumer_disc",
    "MAT": "consumer_disc",
    "HAS": "consumer_disc",
    "NWL": "consumer_disc",
    "COST": "consumer_disc",
    "ROST": "consumer_disc",
    "PEP": "consumer_staples",
    "KMB": "consumer_staples",
    # Financials
    "FITB": "financial",
    "NTRS": "financial",
    # Utilities
    "AEP": "utility",
    # Healthcare
    "AMGN": "healthcare",
    # Tech / Media
    "EA": "tech_media",
}

# Sectors that should get oil-correlated features
OIL_CORR_SECTORS = {
    "energy", "industrial", "consumer_disc", "consumer_staples",
}

# Sectors that should get rate-correlated features
RATE_CORR_SECTORS = {
    "financial", "utility", "healthcare", "tech_media",
}


def sector_for(ticker: str) -> str:
    return TICKER_SECTOR[ticker]


def feature_recipe(ticker: str) -> list[str]:
    """Return list of sector-aware feature column names for this ticker."""
    sector = sector_for(ticker)
    if sector in OIL_CORR_SECTORS:
        return ["stock_oil_corr_21", "vol_x_oil"]
    if sector in RATE_CORR_SECTORS:
        return ["stock_rate_corr_21"]
    raise ValueError(f"Unknown sector mapping for {ticker}: {sector}")


if __name__ == "__main__":
    for tkr in sorted(TICKER_SECTOR.keys()):
        print(f"{tkr:6s} -> {sector_for(tkr):16s} feats={feature_recipe(tkr)}")
