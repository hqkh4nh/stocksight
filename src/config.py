from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config() -> dict:
    """Read config.yaml and return as dict."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()

# Quick-access path helpers
DATA_DIR = PROJECT_ROOT / CFG["data"]["cache_dir"]
MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models_dir"]
RESULTS_DIR = PROJECT_ROOT / CFG["paths"]["results_dir"]

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    # Test: run this file directly to verify config loads
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("Tickers:", CFG["tickers"])