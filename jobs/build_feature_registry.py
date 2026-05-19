from __future__ import annotations

from _common import ROOT
from home_credit_risk.config import load_config
from home_credit_risk.features.bureau import build_bureau_features
from home_credit_risk.features.bureau_balance import build_bureau_balance_features
from home_credit_risk.features.credit_card_balance import build_credit_card_features
from home_credit_risk.features.installments_payments import build_installments_features
from home_credit_risk.features.pos_cash_balance import build_pos_cash_features
from home_credit_risk.features.previous_application import build_previous_application_features
from home_credit_risk.io import read_csv_table
from home_credit_risk.logger import setup_logger
from home_credit_risk.paths import build_paths
from home_credit_risk.registry import registry_to_dataframe, write_registry
from home_credit_risk.spark_session import build_spark_session


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    paths = build_paths(config)
    logger = setup_logger("build_feature_registry", paths.logs)
    spark = build_spark_session(config)

    bureau = read_csv_table(spark, paths.raw / config.raw["tables"]["bureau"])
    bureau_balance = read_csv_table(spark, paths.raw / config.raw["tables"]["bureau_balance"])
    previous = read_csv_table(spark, paths.raw / config.raw["tables"]["previous_application"])
    installments = read_csv_table(spark, paths.raw / config.raw["tables"]["installments_payments"])
    pos_cash = read_csv_table(spark, paths.raw / config.raw["tables"]["pos_cash_balance"])
    credit_card = read_csv_table(spark, paths.raw / config.raw["tables"]["credit_card_balance"])

    bb_features, bb_registry = build_bureau_balance_features(bureau_balance)
    bureau_features, bureau_registry = build_bureau_features(bureau, bb_features)
    prev_features, prev_registry = build_previous_application_features(previous)
    inst_features, inst_registry = build_installments_features(installments)
    pos_features, pos_registry = build_pos_cash_features(pos_cash)
    cc_features, cc_registry = build_credit_card_features(credit_card)

    registry_df = registry_to_dataframe(
        spark,
        [*bb_registry, *bureau_registry, *prev_registry, *inst_registry, *pos_registry, *cc_registry],
    )

    write_registry(registry_df, paths.registry / "feature_registry")
    logger.info("Wrote feature registry to %s", paths.registry)


if __name__ == "__main__":
    main()