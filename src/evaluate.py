"""Backwards-compatible shim. Real implementation in src.evaluation_report."""
from src.evaluation_report import dl_extended_metrics


def dl_metrics_14d(model, dl) -> dict:
    """Legacy entry point. Calls extended metrics with no history attached."""
    return dl_extended_metrics(model, dl, history=None)
