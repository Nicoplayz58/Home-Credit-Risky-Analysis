from __future__ import annotations

from _common import ROOT
from home_credit_risk.config import load_config
from home_credit_risk.features.bureau import build_bureau_features
from home_credit_risk.features.bureau_balance import build_bureau_balance_features
from home_credit_risk.io import read_csv_table, write_parquet
from home_credit_risk.logger import setup_logger
from home_credit_risk.paths import build_paths
from home_credit_risk.spark_session import build_spark_session
from home_credit_risk.validation import assert_unique_keys


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    paths = build_paths(config)
    logger = setup_logger("build_bureau", paths.logs)
    spark = build_spark_session(config)

    bureau = read_csv_table(spark, paths.raw / config.raw["tables"]["bureau"])
    bureau_balance = read_csv_table(spark, paths.raw / config.raw["tables"]["bureau_balance"])

    bureau_balance_features, _ = build_bureau_balance_features(bureau_balance)
    features, _ = build_bureau_features(bureau, bureau_balance_features)

    result = assert_unique_keys(features, ["SK_ID_CURR"], "bureau_features")
    if not result.passed:
        raise ValueError(result.details)

    output_path = paths.interim / "bureau_features"
    write_parquet(features, output_path)
    logger.info("Wrote bureau features to %s", output_path)


if __name__ == "__main__":
    main()