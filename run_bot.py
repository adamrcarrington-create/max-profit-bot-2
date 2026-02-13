#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from kalshi_mm.config import BotConfig
from kalshi_mm.kalshi_api import KalshiApiError, auth_diagnostics, resolve_credentials
from kalshi_mm.runner import (
    run_paper,
    run_paper_multi,
    verdict,
    write_multi_report,
    write_report,
)
from kalshi_mm.sandbox_bot import run_sandbox


def parse_args() -> argparse.Namespace:
    default_report = str((Path(__file__).resolve().parent / "reports" / "latest_report.json"))
    p = argparse.ArgumentParser(
        description="Risk-first Kalshi-style market maker (paper mode + Kalshi sandbox mode)."
    )
    p.add_argument("--mode", choices=["paper", "sandbox", "live"], default="paper")
    p.add_argument("--config", type=str, default=None, help="Optional JSON config file")
    p.add_argument("--market", type=str, default=None, help="Override market ticker")
    p.add_argument(
        "--markets",
        type=str,
        default=None,
        help="Comma-separated tickers (example: KXBTC15M,KXETH15M,KXSOL15M)",
    )
    p.add_argument("--steps", type=int, default=None, help="Simulation steps per episode")
    p.add_argument("--episodes", type=int, default=50, help="Monte Carlo episodes")
    p.add_argument("--seed", type=int, default=7, help="Base RNG seed")
    p.add_argument("--target-return", type=float, default=None, help="Target median return pct")
    p.add_argument("--max-drawdown", type=float, default=None, help="Max p90 drawdown pct")
    p.add_argument("--optimize", action="store_true", help="Run strategy parameter search first")
    p.add_argument(
        "--report",
        type=str,
        default=default_report,
        help="Output JSON report path",
    )
    p.add_argument(
        "--allow-live",
        action="store_true",
        help="Required to attempt live mode (still blocked in this build).",
    )
    p.add_argument("--api-key-id", type=str, default=None, help="Kalshi API key id. If omitted, reads KALSHI_API_KEY_ID.")
    p.add_argument(
        "--private-key-path",
        type=str,
        default=None,
        help="Path to Kalshi RSA private key PEM. If omitted, reads KALSHI_PRIVATE_KEY_PATH.",
    )
    p.add_argument(
        "--api-base-url",
        type=str,
        default="https://demo-api.kalshi.co",
        help="Kalshi API base URL. Demo default is https://demo-api.kalshi.co",
    )
    p.add_argument("--loop-seconds", type=float, default=8.0, help="Sandbox loop interval seconds.")
    p.add_argument("--runtime-minutes", type=float, default=20.0, help="Sandbox runtime duration in minutes.")
    p.add_argument(
        "--pss-salt-mode",
        choices=["digest", "max"],
        default="digest",
        help="Signature salt mode for RSA-PSS (for auth compatibility).",
    )
    p.add_argument(
        "--submit-orders",
        action="store_true",
        help="Sandbox mode: actually submit/cancel orders. If not set, runs dry-run only.",
    )
    p.add_argument(
        "--auth-diagnose",
        action="store_true",
        help="Run authentication diagnostics across known Kalshi base URLs; does not place orders.",
    )
    return p.parse_args()


def _print_auth_help(base_url: str) -> None:
    print("Authentication failed (401). Most likely causes:")
    print("1) API key ID is not the key UUID (using key name instead of key id).")
    print("2) API key ID and private key file are not a matching pair.")
    print("3) Environment mismatch:")
    print("   - Demo key must use https://demo-api.kalshi.co")
    print("   - Production key must use https://api.elections.kalshi.com")
    print(f"Current base_url={base_url}")
    print("Try creating a fresh key in the same environment and rerun in dry-run first.")


def main() -> int:
    args = parse_args()

    cfg = BotConfig.from_json(args.config) if args.config else BotConfig()
    if args.market:
        cfg.market_ticker = args.market.upper().strip()
        cfg.market_tickers = [cfg.market_ticker]
    if args.markets:
        cfg.market_tickers = [
            m.strip().upper() for m in args.markets.split(",") if m and m.strip()
        ]
        if cfg.market_tickers:
            cfg.market_ticker = cfg.market_tickers[0]
    if args.steps is not None:
        cfg.sim.steps = args.steps
    if args.target_return is not None:
        cfg.target_return_pct = args.target_return
    if args.max_drawdown is not None:
        cfg.max_drawdown_pct = args.max_drawdown

    markets = cfg.resolved_markets()

    if args.auth_diagnose:
        try:
            creds = resolve_credentials(args.api_key_id, args.private_key_path)
        except ValueError as exc:
            print(str(exc))
            return 2
        results = auth_diagnostics(credentials=creds)
        print("--- auth_diagnostics ---")
        for r in results:
            print(
                f"base_url={r['base_url']} "
                f"pss_salt_mode={r['pss_salt_mode']} "
                f"status={r['status']}"
            )
            print(f"detail={r['detail']}")
        print("If every row is 401, regenerate a fresh API key pair in the intended environment.")
        return 0

    if args.mode == "sandbox":
        try:
            creds = resolve_credentials(args.api_key_id, args.private_key_path)
        except ValueError as exc:
            print(str(exc))
            return 2
        try:
            report = run_sandbox(
                cfg=cfg,
                credentials=creds,
                markets=markets,
                loop_seconds=args.loop_seconds,
                runtime_minutes=args.runtime_minutes,
                submit_orders=args.submit_orders,
                report_path=args.report,
                base_url=args.api_base_url,
                pss_salt_mode=args.pss_salt_mode,
            )
        except KalshiApiError as exc:
            err_text = str(exc)
            print(f"sandbox_error={err_text}")
            if "status=401" in err_text or "authentication_error" in err_text:
                _print_auth_help(args.api_base_url)
            return 4

        requested = report.get("requested_markets", markets)
        resolved = report.get("resolved_markets", markets)
        print(
            "mode=sandbox "
            f"requested_markets={','.join(requested)} "
            f"resolved_markets={','.join(resolved)}"
        )
        print(f"submit_orders={args.submit_orders}")
        print(f"base_url={args.api_base_url}")
        print(f"pss_salt_mode={args.pss_salt_mode}")
        print("--- runtime ---")
        runtime = report.get("runtime", {})
        counts = report.get("counts", {})
        risk_controls = report.get("risk_controls", {})
        print(f"cycles={runtime.get('cycles', 0)}")
        print(f"market_rolls={runtime.get('market_rolls', 0)}")
        print(f"elapsed_seconds={runtime.get('elapsed_seconds', 0.0):.1f}")
        print(f"stop_reason={runtime.get('stop_reason', 'unknown')}")
        print(f"session_stop_loss_cents={risk_controls.get('session_stop_loss_cents', 0)}")
        print(f"session_take_profit_cents={risk_controls.get('session_take_profit_cents', 0)}")
        print(f"session_drawdown_cents={risk_controls.get('session_drawdown_cents', 0)}")
        print(f"preclose_guard_seconds={risk_controls.get('preclose_guard_seconds', 0)}")
        print(
            "no_new_entries_before_close_seconds="
            f"{risk_controls.get('no_new_entries_before_close_seconds', 0)}"
        )
        print(
            "no_trade_window_cycle_limit="
            f"{risk_controls.get('no_trade_window_cycle_limit', 0)}"
        )
        print(
            "auto_roll_on_no_trade_window="
            f"{risk_controls.get('auto_roll_on_no_trade_window', False)}"
        )
        print(f"momentum_soft_guard={risk_controls.get('momentum_soft_guard', 0)}")
        print(f"momentum_hard_guard={risk_controls.get('momentum_hard_guard', 0)}")
        print(f"adverse_side_size_cut={risk_controls.get('adverse_side_size_cut', 0)}")
        print(f"toxicity_edge_boost={risk_controls.get('toxicity_edge_boost', 0)}")
        print(f"min_requote_seconds={risk_controls.get('min_requote_seconds', 0)}")
        print(f"max_orders_per_market={risk_controls.get('max_orders_per_market', 0)}")
        print(f"min_market_spread={risk_controls.get('min_market_spread', 0)}")
        print(f"min_quote_edge={risk_controls.get('min_quote_edge', 0)}")
        print(f"orders_submitted={counts.get('orders_submitted', 0)}")
        print(f"orders_canceled={counts.get('orders_canceled', 0)}")
        print(f"orders_canceled_on_exit={counts.get('orders_canceled_on_exit', 0)}")
        print(f"flatten_orders_submitted={counts.get('flatten_orders_submitted', 0)}")
        print(f"preclose_flatten_warnings={counts.get('preclose_flatten_warnings', 0)}")
        print(f"flatten_errors={counts.get('flatten_errors', 0)}")
        print(f"flatten_residual_markets={counts.get('flatten_residual_markets', 0)}")
        print(f"errors={counts.get('errors', 0)}")
        for note in report.get("market_resolution_notes", []):
            print(
                "resolution "
                f"requested={note.get('requested', '')} "
                f"resolved={note.get('resolved', '')} "
                f"method={note.get('method', '')}"
            )
        for m in report.get("market_summaries", []):
            print(f"--- market {m.get('ticker', 'UNKNOWN')} ---")
            print(f"cycles={m.get('cycles', 0)}")
            print(f"orders_submitted={m.get('orders_submitted', 0)}")
            print(f"orders_canceled={m.get('orders_canceled', 0)}")
            print(f"post_only_reprices={m.get('post_only_reprices', 0)}")
            print(f"post_only_rejects={m.get('post_only_rejects', 0)}")
            print(f"preclose_flatten_calls={m.get('preclose_flatten_calls', 0)}")
            print(f"max_abs_position={m.get('max_abs_position', 0)}")
            print(f"final_position={m.get('final_position', 0)}")
            print(f"last_reason={m.get('last_reason', '')}")
        print(f"report={Path(args.report).resolve()}")
        return 0

    if args.mode == "live":
        if not args.allow_live:
            print("Refusing live mode without --allow-live.")
            return 2
        print("Live production mode is blocked in this build.")
        print("Use --mode sandbox with demo-api first.")
        return 3

    if len(markets) > 1:
        market_results, combined = run_paper_multi(
            cfg=cfg,
            markets=markets,
            episodes=args.episodes,
            seed=args.seed,
            optimize=args.optimize,
        )
        write_multi_report(
            file_path=args.report,
            cfg=cfg,
            market_results=market_results,
            combined=combined,
        )
        print(f"markets={','.join(markets)}")
        print(f"mode=paper episodes={args.episodes} steps={cfg.sim.steps}")
        print(f"target_return_pct={cfg.target_return_pct:.3f}")
        print(f"max_drawdown_pct={cfg.max_drawdown_pct:.3f}")
        print("--- combined_summary ---")
        print(f"market_count={int(combined['market_count'])}")
        print(f"median_return_pct_avg={combined['median_return_pct_avg']:.4f}")
        print(f"p10_return_pct_worst={combined['p10_return_pct_worst']:.4f}")
        print(f"max_drawdown_p90_pct_worst={combined['max_drawdown_p90_pct_worst']:.4f}")
        print(f"kill_rate_worst={combined['kill_rate_worst']:.4f}")
        print(f"pass_ratio={combined['pass_ratio']:.4f}")
        print(
            "per_market_limits="
            f"max_position:{int(combined['per_market_max_position'])},"
            f"max_order_size:{int(combined['per_market_max_order_size'])}"
        )
        for result in market_results:
            s = result.summary
            print(f"--- market {result.market} ---")
            print(f"median_return_pct={s.median_return_pct:.4f}")
            print(f"p10_return_pct={s.p10_return_pct:.4f}")
            print(f"p90_return_pct={s.p90_return_pct:.4f}")
            print(f"max_drawdown_p90_pct={s.max_drawdown_p90_pct:.4f}")
            print(f"kill_rate={s.kill_rate:.4f}")
            print(f"avg_pnl={s.avg_pnl:.4f}")
            print(f"avg_trades={s.avg_trades:.2f}")
            print(f"avg_win_rate={s.avg_win_rate:.4f}")
            print(f"go_no_go={result.go_no_go}")
            if result.optimization:
                print("--- optimized_strategy ---")
                print(json.dumps(asdict(result.strategy), indent=2))
        print(f"report={Path(args.report).resolve()}")
        return 0

    strategy, summary, metrics, optimization = run_paper(
        cfg=cfg,
        episodes=args.episodes,
        seed=args.seed,
        optimize=args.optimize,
    )

    go_no_go = verdict(
        summary=summary,
        target_return_pct=cfg.target_return_pct,
        max_drawdown_pct=cfg.max_drawdown_pct,
    )

    write_report(
        file_path=args.report,
        cfg=cfg,
        strategy=strategy,
        summary=summary,
        metrics=metrics,
        optimization=optimization,
    )

    print(f"market={cfg.market_ticker}")
    print(f"mode=paper episodes={summary.episodes} steps={cfg.sim.steps}")
    print(f"target_return_pct={cfg.target_return_pct:.3f}")
    print(f"max_drawdown_pct={cfg.max_drawdown_pct:.3f}")
    print("--- summary ---")
    print(f"median_return_pct={summary.median_return_pct:.4f}")
    print(f"p10_return_pct={summary.p10_return_pct:.4f}")
    print(f"p90_return_pct={summary.p90_return_pct:.4f}")
    print(f"max_drawdown_p90_pct={summary.max_drawdown_p90_pct:.4f}")
    print(f"kill_rate={summary.kill_rate:.4f}")
    print(f"avg_pnl={summary.avg_pnl:.4f}")
    print(f"avg_trades={summary.avg_trades:.2f}")
    print(f"avg_win_rate={summary.avg_win_rate:.4f}")
    print(f"go_no_go={go_no_go}")

    if optimization:
        print("--- optimized_strategy ---")
        print(json.dumps(asdict(strategy), indent=2))

    print(f"report={Path(args.report).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
