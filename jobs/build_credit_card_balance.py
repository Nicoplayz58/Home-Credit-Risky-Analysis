from __future__ import annotations

from _common import ROOT
from home_credit_risk.config import load_config
from home_credit_risk.features.credit_card_balance import build_credit_card_features
from home_credit_risk.io import read_csv_table, write_parquet
from home_credit_risk.logger import setup_logger
from home_credit_risk.paths import build_paths
from home_credit_risk.spark_session import build_spark_session
from home_credit_risk.validation import assert_unique_keys


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    paths = build_paths(config)
    logger = setup_logger("build_credit_card_balance", paths.logs)
    spark = build_spark_session(config)

    credit_card = read_csv_table(spark, paths.raw / config.raw["tables"]["credit_card_balance"])
    features, _ = build_credit_card_features(credit_card)

    result = assert_unique_keys(features, ["SK_ID_CURR"], "credit_card_features")
    if not result.passed:
        raise ValueError(result.details)

    output_path = paths.interim / "credit_card_features"
    write_parquet(features, output_path)
    logger.info("Wrote credit card features to %s", output_path)


if __name__ == "__main__":
    main()