from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from kalshi_hft.backtest import implied_yes_probability, load_rows
from kalshi_hft.client import extract_float
from kalshi_hft.engine import BotConfig, clamp
from kalshi_hft.client import KalshiClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate real_yes probability blend against settled market outcomes."
    )
    parser.add_argument(
        "--inputs-glob",
        type=str,
        default="logs/replay_capture*.jsonl",
        help="Glob for replay files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.example.json"),
        help="Config JSON for Kalshi auth + strategy params.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/signal_calibration_report.json",
        help="Output report path.",
    )
    parser.add_argument(
        "--max-rows-per-market",
        type=int,
        default=200,
        help="Cap rows sampled per replay file.",
    )
    return parser.parse_args()


def _split_symbol_direction(row0: dict[str, Any], market: dict[str, Any]) -> str:
    direction = str(row0.get("direction") or "").strip().lower()
    if direction in {"up", "down"}:
        return direction
    title = str(market.get("title") or "").lower()
    yes_sub = str(market.get("yes_sub_title") or "").lower()
    if any(k in title for k in ["down", "below", "under"]) or any(k in yes_sub for k in ["below", "under"]):
        return "down"
    return "up"


def _market_label(market: dict[str, Any]) -> Optional[int]:
    result = str(market.get("result") or "").strip().lower()
    if result in {"yes", "1", "true"}:
        return 1
    if result in {"no", "0", "false"}:
        return 0
    return None


def _extract_strike(row0: dict[str, Any], market: dict[str, Any]) -> Optional[float]:
    return extract_float(
        row0.get("strike")
        or row0.get("strike_price")
        or row0.get("target")
        or row0.get("target_price")
        or market.get("strike")
        or market.get("strike_price")
        or market.get("target")
        or market.get("target_price")
        or market.get("floor_strike")
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _real_yes(
    *,
    spot: float,
    strike: float,
    direction: str,
    momentum_delta: float,
    distance_scale_pct: float,
    momentum_scale_dollars: float,
    distance_weight: float,
) -> float:
    strike_ref = strike if strike > 0 else spot
    dist_scale = max(25.0, abs(strike_ref) * distance_scale_pct)
    z = (spot - strike_ref) / dist_scale
    if direction == "down":
        z = -z
    dist_prob = _sigmoid(z)
    momentum_move = clamp(momentum_delta / max(1.0, momentum_scale_dollars), -0.5, 0.5)
    if direction == "down":
        momentum_move = -momentum_move
    momentum_prob = 0.5 + momentum_move
    w = clamp(distance_weight, 0.0, 1.0)
    p = w * dist_prob + (1.0 - w) * momentum_prob
    return clamp(p, 0.01, 0.99)


def _brier(pred: float, label: int) -> float:
    return (pred - float(label)) ** 2


def _market_dict(payload: dict[str, Any]) -> dict[str, Any]:
    market = payload.get("market")
    if isinstance(market, dict):
        return market
    return payload


def main() -> int:
    args = parse_args()
    cfg = BotConfig.from_json(args.config)
    client = KalshiClient(cfg.kalshi)

    files = sorted(Path().glob(args.inputs_glob))
    if not files:
        raise RuntimeError(f"no files matched glob={args.inputs_glob}")

    samples: list[dict[str, Any]] = []
    used_files: list[str] = []
    skipped_files: list[dict[str, str]] = []

    for path in files:
        rows = load_rows(path)
        if not rows:
            skipped_files.append({"file": str(path), "reason": "no_rows"})
            continue
        row0 = rows[0]
        ticker = str(row0.get("ticker") or "").strip()
        if not ticker:
            skipped_files.append({"file": str(path), "reason": "missing_ticker"})
            continue
        try:
            market = _market_dict(client.get_market(ticker))
        except Exception as exc:
            skipped_files.append({"file": str(path), "reason": f"market_fetch_error:{exc}"})
            continue
        label = _market_label(market)
        if label is None:
            skipped_files.append({"file": str(path), "reason": "market_not_settled_or_result_missing"})
            continue
        strike = _extract_strike(row0, market)
        if strike is None:
            skipped_files.append({"file": str(path), "reason": "missing_strike"})
            continue
        direction = _split_symbol_direction(row0, market)

        # Build simple momentum over trailing rows in this replay.
        rows_sorted = sorted(
            [r for r in rows if extract_float(r.get("ref_price") or r.get("spot_price")) is not None],
            key=lambda r: int(r.get("timestamp_ms") or 0),
        )
        if not rows_sorted:
            skipped_files.append({"file": str(path), "reason": "no_ref_price"})
            continue
        step = max(1, len(rows_sorted) // max(1, args.max_rows_per_market))
        for idx in range(0, len(rows_sorted), step):
            row = rows_sorted[idx]
            spot = extract_float(row.get("ref_price") or row.get("spot_price"))
            implied = implied_yes_probability(row)
            if spot is None or implied is None:
                continue
            lookback_idx = max(0, idx - 10)
            baseline = rows_sorted[lookback_idx]
            baseline_spot = extract_float(baseline.get("ref_price") or baseline.get("spot_price"))
            if baseline_spot is None:
                baseline_spot = spot
            momentum_delta = float(spot - baseline_spot)
            samples.append(
                {
                    "ticker": ticker,
                    "spot": float(spot),
                    "implied": float(implied),
                    "strike": float(strike),
                    "direction": direction,
                    "momentum_delta": momentum_delta,
                    "label": int(label),
                }
            )
        used_files.append(str(path))

    if not samples:
        raise RuntimeError("no settled samples available; capture older settled markets and rerun")

    distance_weights = [0.3, 0.5, 0.7, 0.85]
    momentum_scales = [50.0, 75.0, 100.0, 150.0, 200.0]
    distance_scale_pcts = [0.0010, 0.0015, 0.0020, 0.0030]

    best: Optional[dict[str, Any]] = None
    leaderboard: list[dict[str, Any]] = []

    implied_brier = sum(_brier(s["implied"], s["label"]) for s in samples) / len(samples)

    for w in distance_weights:
        for mom_scale in momentum_scales:
            for dist_scale in distance_scale_pcts:
                total_brier = 0.0
                for s in samples:
                    pred = _real_yes(
                        spot=s["spot"],
                        strike=s["strike"],
                        direction=s["direction"],
                        momentum_delta=s["momentum_delta"],
                        distance_scale_pct=dist_scale,
                        momentum_scale_dollars=mom_scale,
                        distance_weight=w,
                    )
                    total_brier += _brier(pred, s["label"])
                brier = total_brier / len(samples)
                row = {
                    "distance_weight": w,
                    "momentum_scale_dollars": mom_scale,
                    "distance_scale_pct": dist_scale,
                    "brier_score": round(brier, 8),
                    "improvement_vs_implied_brier": round(implied_brier - brier, 8),
                }
                leaderboard.append(row)
                if best is None or brier < best["brier_score"]:
                    best = row

    leaderboard.sort(key=lambda r: float(r["brier_score"]))
    assert best is not None

    best_model_params = dict(best)
    blend_candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    blend_rows: list[dict[str, Any]] = []
    best_blend: Optional[dict[str, Any]] = None
    for mult in blend_candidates:
        total_brier = 0.0
        for s in samples:
            model_pred = _real_yes(
                spot=s["spot"],
                strike=s["strike"],
                direction=s["direction"],
                momentum_delta=s["momentum_delta"],
                distance_scale_pct=float(best_model_params["distance_scale_pct"]),
                momentum_scale_dollars=float(best_model_params["momentum_scale_dollars"]),
                distance_weight=float(best_model_params["distance_weight"]),
            )
            blended = clamp(s["implied"] + (model_pred - s["implied"]) * mult, 0.01, 0.99)
            total_brier += _brier(blended, s["label"])
        brier = total_brier / len(samples)
        row = {
            "model_edge_multiplier": mult,
            "brier_score": round(brier, 8),
            "improvement_vs_implied_brier": round(implied_brier - brier, 8),
        }
        blend_rows.append(row)
        if best_blend is None or brier < float(best_blend["brier_score"]):
            best_blend = row

    out_payload = {
        "samples": len(samples),
        "files_used": used_files,
        "files_skipped": skipped_files,
        "baseline_implied_brier": round(implied_brier, 8),
        "best": best,
        "best_blend": best_blend,
        "top10": leaderboard[:10],
        "blend_grid": blend_rows,
        "suggested_strategy_overrides": {
            "distance_scale_pct": best["distance_scale_pct"],
            "momentum_full_scale_dollars": best["momentum_scale_dollars"],
            "distance_weight_hint": best["distance_weight"],
            "momentum_weight_hint": round(1.0 - float(best["distance_weight"]), 4),
            "model_edge_multiplier": best_blend["model_edge_multiplier"] if best_blend else 1.0,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2))
    print(json.dumps(out_payload, indent=2))
    print(f"report={out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
