from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

PIPELINE_STEPS = [
    ("build_bureau_balance", ROOT / "jobs" / "build_bureau_balance.py"),
    ("build_bureau", ROOT / "jobs" / "build_bureau.py"),
    ("build_previous_application", ROOT / "jobs" / "build_previous_application.py"),
    ("build_installments_payments", ROOT / "jobs" / "build_installments_payments.py"),
    ("build_pos_cash_balance", ROOT / "jobs" / "build_pos_cash_balance.py"),
    ("build_credit_card_balance", ROOT / "jobs" / "build_credit_card_balance.py"),
    ("build_feature_registry", ROOT / "jobs" / "build_feature_registry.py"),
    ("build_train_dataset", ROOT / "build_train_dataset.py"),
    ("data_quality_checks", ROOT / "jobs" / "data_quality_checks.py"),
    ("drift_monitoring", ROOT / "jobs" / "drift_monitoring.py"),
]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def run_step(logger: logging.Logger, name: str, script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    logger.info("Starting step: %s", name)
    python_executable = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    env = os.environ.copy()
    env["PYSPARK_PYTHON"] = python_executable
    env["PYSPARK_DRIVER_PYTHON"] = python_executable
    result = subprocess.run([python_executable, str(script_path)], cwd=str(ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (exit code {result.returncode})")
    logger.info("Completed step: %s", name)


def main() -> int:
    logger = setup_logger()
    logger.info("Pipeline execution started")

    try:
        for name, script_path in PIPELINE_STEPS:
            run_step(logger, name, script_path)
    except Exception as exc:
        logger.error("Pipeline stopped: %s", exc)
        return 1

    logger.info("Pipeline execution finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())