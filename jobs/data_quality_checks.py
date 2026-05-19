from __future__ import annotations

from _common import ROOT
from home_credit_risk.config import load_config
from home_credit_risk.io import read_csv_table
from home_credit_risk.logger import setup_logger
from home_credit_risk.paths import build_paths
from home_credit_risk.quality import basic_profile
from home_credit_risk.spark_session import build_spark_session


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    paths = build_paths(config)
    logger = setup_logger("data_quality_checks", paths.logs)
    spark = build_spark_session(config)

    train = read_csv_table(spark, paths.raw / config.raw["tables"]["application_train"])
    test = read_csv_table(spark, paths.raw / config.raw["tables"]["application_test"])

    logger.info("Train rows: %s", train.count())
    logger.info("Test rows: %s", test.count())
    logger.info("Train columns: %s", len(train.columns))
    logger.info("Test columns: %s", len(test.columns))

    train_profile = basic_profile(train)
    test_profile = basic_profile(test)
    train_profile.write.mode("overwrite").parquet(str(paths.checks / "train_application_profile.parquet"))
    test_profile.write.mode("overwrite").parquet(str(paths.checks / "test_application_profile.parquet"))
    logger.info("Saved basic profile checks to %s", paths.checks)


if __name__ == "__main__":
    main()