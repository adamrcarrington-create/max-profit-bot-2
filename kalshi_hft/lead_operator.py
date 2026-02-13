from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional


class LeadTradingOperator:
    """
    Supervises kalshi_bot.py:
    - Restarts on persistent 401 unauthorized failures.
    - Emits performance summaries every 15 minutes from performance.json.
    - Stops fully if kill-switch exit is detected.
    """

    def __init__(
        self,
        *,
        config_path: str,
        performance_json: str,
        operator_log: str,
        restart_delay_seconds: float = 2.0,
        max_restarts_per_hour: int = 20,
    ) -> None:
        self.config_path = config_path
        self.performance_json = performance_json
        self.operator_log = Path(operator_log)
        self.operator_log.parent.mkdir(parents=True, exist_ok=True)
        self.restart_delay_seconds = max(0.5, restart_delay_seconds)
        self.max_restarts_per_hour = max(1, max_restarts_per_hour)
        self._restarts: list[float] = []
        self._next_performance_print = 0.0
        self.proc: Optional[subprocess.Popen[str]] = None

    def run(self) -> int:
        self._log("operator_start")
        self._spawn()
        self._next_performance_print = time.time() + 900.0
        try:
            while True:
                if self.proc is None:
                    self._spawn()
                assert self.proc is not None

                line = self.proc.stdout.readline() if self.proc.stdout else ""
                if line:
                    clean = line.rstrip("\n")
                    print(clean)
                    self._log(f"bot_log {clean}")
                    if self._is_unauthorized_line(clean):
                        self._log("detected_401 restarting_auth_sequence")
                        if not self._restart():
                            self._log("restart_rate_limit_hit stopping_operator")
                            return 5

                code = self.proc.poll()
                if code is not None:
                    self._log(f"bot_exit code={code}")
                    if code == 12:
                        self._log("kill_switch_exit_detected not_restarting")
                        return 12
                    if not self._restart():
                        self._log("restart_rate_limit_hit stopping_operator")
                        return 6

                now = time.time()
                if now >= self._next_performance_print:
                    self._print_performance_summary()
                    self._next_performance_print = now + 900.0
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._log("operator_keyboard_interrupt")
            self._stop_bot()
            return 0

    def _spawn(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            "-m",
            "kalshi_hft.kalshi_bot",
            "--config",
            self.config_path,
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log(f"spawned pid={self.proc.pid}")

    def _restart(self) -> bool:
        now = time.time()
        self._restarts = [ts for ts in self._restarts if ts >= now - 3600.0]
        if len(self._restarts) >= self.max_restarts_per_hour:
            return False
        self._restarts.append(now)
        self._stop_bot()
        time.sleep(self.restart_delay_seconds)
        self._spawn()
        return True

    def _stop_bot(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
        self.proc = None

    def _print_performance_summary(self) -> None:
        perf_path = Path(self.performance_json)
        if not perf_path.exists():
            msg = "performance_summary unavailable file_missing"
            print(msg)
            self._log(msg)
            return
        try:
            payload = json.loads(perf_path.read_text())
        except Exception as exc:
            msg = f"performance_summary parse_error={exc}"
            print(msg)
            self._log(msg)
            return
        wins = payload.get("wins", 0)
        losses = payload.get("losses", 0)
        slippage = payload.get("slippage_cents", 0.0)
        ticker = payload.get("ticker", "UNKNOWN")
        as_of = payload.get("as_of", "UNKNOWN")
        msg = (
            f"performance_summary as_of={as_of} ticker={ticker} "
            f"wins={wins} losses={losses} slippage_cents={slippage}"
        )
        print(msg)
        self._log(msg)

    def _is_unauthorized_line(self, line: str) -> bool:
        low = line.lower()
        return ("401" in low and "unauthorized" in low) or ("status=401" in low)

    def _log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n"
        with self.operator_log.open("a", encoding="utf-8") as f:
            f.write(line)


def parse_args() -> argparse.Namespace:
    default_config = str(Path(__file__).resolve().parent / "config.example.json")
    default_perf = str(Path(__file__).resolve().parents[1] / "logs" / "performance.json")
    default_operator_log = str(Path(__file__).resolve().parents[1] / "logs" / "operator.log")
    parser = argparse.ArgumentParser(description="Lead Trading Operator for kalshi_bot.py")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--performance-json", type=str, default=default_perf)
    parser.add_argument("--operator-log", type=str, default=default_operator_log)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operator = LeadTradingOperator(
        config_path=args.config,
        performance_json=args.performance_json,
        operator_log=args.operator_log,
    )
    return operator.run()


if __name__ == "__main__":
    raise SystemExit(main())
