"""
IPPO vs MAPPO A/B Test Validation - Phase 0 Decision Gate

Compares IPPO+global state baseline against MAPPO (centralized critic).
Decision gate: proceed to Phase 1 only if MAPPO shows 15%+ improvement.

Usage:
    # Quick validation (1K steps each)
    python tests/validation_ab_test.py --steps 1000

    # Full validation (10K steps each)
    python tests/validation_ab_test.py --steps 10000
"""

import argparse
import json
import logging
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


VALIDATION_THRESHOLD = 15.0  # Minimum improvement % for MAPPO to proceed
OUTPUT_DIR = Path("outputs")


class ABTestRunner:
    def __init__(
        self,
        steps: int = 10000,
        num_eval_episodes: int = 50,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.steps = steps
        self.num_eval_episodes = num_eval_episodes
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_ippo_global(self) -> Dict:
        logger.info(f"Running IPPO+global baseline for {self.steps} steps...")
        start_time = time.time()

        log_dir = (
            self.output_dir / f"ippo_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        train_cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "train" / "train_ippo_global.py"),
            "--steps",
            str(self.steps),
            "--n-envs",
            "1",
            "--eval-interval",
            str(max(self.steps // 10, 100)),
            "--log-dir",
            str(log_dir),
        ]

        train_result = subprocess.run(
            train_cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT
        )
        train_duration = time.time() - start_time
        logger.info(f"IPPO+global training completed in {train_duration:.2f}s")

        model_path = log_dir / "ippo_global_model.zip"
        eval_results = self._run_evaluation("ippo", str(model_path))

        duration = time.time() - start_time

        return {
            "win_rate": eval_results.get("win_rate", 0.0),
            "avg_reward": eval_results.get("avg_reward", 0.0),
            "avg_length": eval_results.get("avg_length", 0.0),
            "identity_wins": eval_results.get("identity_wins", {}),
            "train_duration": train_duration,
            "eval_duration": eval_results.get("duration_seconds", 0.0),
            "total_duration": duration,
            "log_dir": str(log_dir),
            "model_path": str(model_path),
            "stdout": train_result.stdout[:500],
            "stderr": train_result.stderr[:500] if train_result.stderr else "",
        }

    def run_mappo(self) -> Dict:
        logger.info(f"Running MAPPO for {self.steps} steps...")
        start_time = time.time()

        log_dir = self.output_dir / f"mappo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        train_cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "train" / "train_mappo.py"),
            "--steps",
            str(self.steps),
            "--n-envs",
            "1",
            "--log-dir",
            str(log_dir),
        ]

        train_result = subprocess.run(
            train_cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT
        )
        train_duration = time.time() - start_time
        logger.info(f"MAPPO training completed in {train_duration:.2f}s")

        model_path = log_dir / "mappo_checkpoint.pt"
        eval_results = self._run_evaluation("mappo", str(model_path))

        duration = time.time() - start_time

        return {
            "win_rate": eval_results.get("win_rate", 0.0),
            "avg_reward": eval_results.get("avg_reward", 0.0),
            "avg_length": eval_results.get("avg_length", 0.0),
            "identity_wins": eval_results.get("identity_wins", {}),
            "train_duration": train_duration,
            "eval_duration": eval_results.get("duration_seconds", 0.0),
            "total_duration": duration,
            "log_dir": str(log_dir),
            "model_path": str(model_path),
            "stdout": train_result.stdout[:500],
            "stderr": train_result.stderr[:500] if train_result.stderr else "",
        }

    def _run_evaluation(self, model_type: str, model_path: str) -> Dict:
        """Run actual game evaluation episodes."""
        eval_cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "train" / "evaluate_model.py"),
            "--model-type",
            model_type,
            "--model-path",
            model_path,
            "--num-episodes",
            str(self.num_eval_episodes),
            "--deterministic",
        ]

        eval_output_path = (
            self.output_dir
            / f"eval_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        eval_cmd.extend(["--output", str(eval_output_path)])

        logger.info(
            f"Running evaluation for {model_type}: {self.num_eval_episodes} episodes"
        )
        eval_result = subprocess.run(
            eval_cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT
        )

        if eval_output_path.exists():
            with open(eval_output_path, "r") as f:
                return json.load(f)

        logger.warning(f"Evaluation output not found, parsing stdout")
        return self._parse_eval_stdout(eval_result.stdout)

    def _parse_eval_stdout(self, stdout: str) -> Dict:
        results = {
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "avg_length": 0.0,
            "identity_wins": {},
        }
        for line in stdout.split("\n"):
            if "Win rate:" in line:
                try:
                    pct_str = line.split("Win rate:")[1].strip().split("%")[0]
                    results["win_rate"] = float(pct_str) / 100.0
                except:
                    pass
            if "Avg reward:" in line:
                try:
                    results["avg_reward"] = float(line.split("Avg reward:")[1].strip())
                except:
                    pass
            if "Avg length:" in line:
                try:
                    parts = line.split("Avg length:")[1].strip().split()
                    results["avg_length"] = float(parts[0])
                except:
                    pass
        return results

    def compute_improvement(self, ippo_result: Dict, mappo_result: Dict) -> float:
        ippo_win = ippo_result.get("win_rate", 0.0)
        mappo_win = mappo_result.get("win_rate", 0.0)

        if ippo_win == 0:
            if mappo_win > 0:
                return 100.0
            return 0.0

        improvement_pct = (mappo_win - ippo_win) / ippo_win * 100
        return improvement_pct

    def decision_gate_check(self, improvement_pct: float) -> str:
        if improvement_pct >= VALIDATION_THRESHOLD:
            return "PROCEED"
        return "ROLLBACK"

    def get_rollback_alternatives(self) -> list:
        return [
            "IPPO+global state (no MAPPO)",
            "Hybrid: MAPPO for 主公+忠臣 only",
            "Delay MAPPO until Phase 2",
        ]

    def run_full_validation(self) -> Dict:
        logger.info("=" * 60)
        logger.info("Phase 0 A/B Test Validation")
        logger.info("=" * 60)

        ippo_result = self.run_ippo_global()
        mappo_result = self.run_mappo()

        improvement_pct = self.compute_improvement(ippo_result, mappo_result)
        decision = self.decision_gate_check(improvement_pct)

        report = {
            "timestamp": datetime.now().isoformat(),
            "steps": self.steps,
            "validation_threshold": VALIDATION_THRESHOLD,
            "ippo_global": ippo_result,
            "mappo": mappo_result,
            "improvement_pct": improvement_pct,
            "decision": decision,
            "rollback_alternatives": self.get_rollback_alternatives()
            if decision == "ROLLBACK"
            else [],
        }

        report_file = self.output_dir / "validation_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"IPPO+global win_rate: {ippo_result['win_rate']:.4f}")
        logger.info(f"MAPPO win_rate: {mappo_result['win_rate']:.4f}")
        logger.info(f"Improvement: {improvement_pct:.2f}%")
        logger.info(f"Threshold: {VALIDATION_THRESHOLD}%")
        logger.info(f"Decision: {decision}")
        logger.info("=" * 60)

        if decision == "ROLLBACK":
            logger.warning("MAPPO did not meet improvement threshold.")
            logger.warning("Consider alternatives:")
            for alt in self.get_rollback_alternatives():
                logger.warning(f"  - {alt}")

        logger.info(f"Report saved to: {report_file}")

        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IPPO vs MAPPO A/B Test Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps for each method",
    )

    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes after training",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for reports",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runner = ABTestRunner(
        steps=args.steps,
        num_eval_episodes=args.num_eval_episodes,
        output_dir=Path(args.output_dir),
    )

    report = runner.run_full_validation()

    if report["decision"] == "PROCEED":
        logger.info("VALIDATION PASSED: Proceeding to Phase 1 (World Model)")
        return 0
    else:
        logger.warning("VALIDATION FAILED: Review alternatives before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
