from __future__ import annotations

from jobs._common import ROOT
from home_credit_risk.config import load_config
from home_credit_risk.features.bureau import build_bureau_features
from home_credit_risk.features.bureau_balance import build_bureau_balance_features
from home_credit_risk.features.credit_card_balance import build_credit_card_features
from home_credit_risk.features.installments_payments import build_installments_features
from home_credit_risk.features.pos_cash_balance import build_pos_cash_features
from home_credit_risk.features.previous_application import build_previous_application_features
from home_credit_risk.io import read_csv_table, write_parquet
from home_credit_risk.logger import setup_logger
from home_credit_risk.paths import build_paths
from home_credit_risk.quality import basic_profile
from home_credit_risk.spark_session import build_spark_session
from home_credit_risk.validation import assert_unique_keys, safe_left_join


def build_master_dataset(spark, config, paths, split_name: str):
    application_file = config.raw["tables"][f"application_{split_name}"]
    application = read_csv_table(spark, paths.raw / application_file)

    bureau = read_csv_table(spark, paths.raw / config.raw["tables"]["bureau"])
    bureau_balance = read_csv_table(spark, paths.raw / config.raw["tables"]["bureau_balance"])
    previous = read_csv_table(spark, paths.raw / config.raw["tables"]["previous_application"])
    installments = read_csv_table(spark, paths.raw / config.raw["tables"]["installments_payments"])
    pos_cash = read_csv_table(spark, paths.raw / config.raw["tables"]["pos_cash_balance"])
    credit_card = read_csv_table(spark, paths.raw / config.raw["tables"]["credit_card_balance"])

    bb_features, _ = build_bureau_balance_features(bureau_balance)
    bureau_features, _ = build_bureau_features(bureau, bb_features)
    prev_features, _ = build_previous_application_features(previous)
    inst_features, _ = build_installments_features(installments)
    pos_features, _ = build_pos_cash_features(pos_cash)
    cc_features, _ = build_credit_card_features(credit_card)

    base_count = application.count()
    base_result = assert_unique_keys(application, ["SK_ID_CURR"], f"application_{split_name}")
    if not base_result.passed:
        raise ValueError(base_result.details)

    master = application
    for name, df in [
        ("bureau_features", bureau_features),
        ("previous_application_features", prev_features),
        ("installments_features", inst_features),
        ("pos_cash_features", pos_features),
        ("credit_card_features", cc_features),
    ]:
        assert_unique_keys(df, ["SK_ID_CURR"], name)
        before = master.count()
        master = safe_left_join(master, df, ["SK_ID_CURR"], name)
        after = master.count()
        if before != after:
            raise ValueError(f"Row count changed after joining {name}: before={before}, after={after}")

    if master.count() != base_count:
        raise ValueError("Final dataset does not preserve application cardinality")

    return master


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    paths = build_paths(config)
    logger = setup_logger("build_train_dataset", paths.logs)
    spark = build_spark_session(config)

    train_final = build_master_dataset(spark, config, paths, "train")
    test_final = build_master_dataset(spark, config, paths, "test")

    write_parquet(train_final, paths.processed / "train_final")
    write_parquet(test_final, paths.processed / "test_final")

    train_profile = basic_profile(train_final)
    test_profile = basic_profile(test_final)
    train_profile.write.mode("overwrite").parquet(str(paths.checks / "train_profile.parquet"))
    test_profile.write.mode("overwrite").parquet(str(paths.checks / "test_profile.parquet"))

    logger.info("Wrote final train/test datasets to %s", paths.processed)


if __name__ == "__main__":
    main()