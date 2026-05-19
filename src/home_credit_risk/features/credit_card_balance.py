from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from ..registry import FeatureSpec


def build_credit_card_features(cc: DataFrame) -> tuple[DataFrame, list[FeatureSpec]]:
    c = cc.select(
        "SK_ID_CURR",
        "SK_ID_PREV",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_CURRENT",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    )
    c = c.withColumn("utilization_ratio", F.when((F.col("AMT_CREDIT_LIMIT_ACTUAL") == 0) | F.col("AMT_CREDIT_LIMIT_ACTUAL").isNull(), F.lit(None)).otherwise(F.col("AMT_BALANCE") / F.col("AMT_CREDIT_LIMIT_ACTUAL")))
    c = c.withColumn("payment_to_min_ratio", F.when((F.col("AMT_INST_MIN_REGULARITY") == 0) | F.col("AMT_INST_MIN_REGULARITY").isNull(), F.lit(None)).otherwise(F.col("AMT_PAYMENT_TOTAL_CURRENT") / F.col("AMT_INST_MIN_REGULARITY")))
    c = c.withColumn("drawing_to_limit_ratio", F.when((F.col("AMT_CREDIT_LIMIT_ACTUAL") == 0) | F.col("AMT_CREDIT_LIMIT_ACTUAL").isNull(), F.lit(None)).otherwise(F.col("AMT_DRAWINGS_CURRENT") / F.col("AMT_CREDIT_LIMIT_ACTUAL")))
    c = c.withColumn("has_dpd", F.when(F.col("SK_DPD") > 0, F.lit(1)).otherwise(F.lit(0)))
    c = c.withColumn("has_dpd_def", F.when(F.col("SK_DPD_DEF") > 0, F.lit(1)).otherwise(F.lit(0)))
    c = c.withColumn("cc_recency", F.abs(F.col("MONTHS_BALANCE")))

    per_prev = (
        c.groupBy("SK_ID_CURR", "SK_ID_PREV")
        .agg(
            F.count(F.lit(1)).alias("cc_months_prev"),
            F.avg("utilization_ratio").alias("cc_utilization_mean_prev"),
            F.max("utilization_ratio").alias("cc_utilization_max_prev"),
            F.avg("payment_to_min_ratio").alias("cc_payment_to_min_mean_prev"),
            F.max("payment_to_min_ratio").alias("cc_payment_to_min_max_prev"),
            F.avg("drawing_to_limit_ratio").alias("cc_drawing_to_limit_mean_prev"),
            F.avg("has_dpd").alias("cc_dpd_rate_prev"),
            F.avg("has_dpd_def").alias("cc_dpd_def_rate_prev"),
            F.max("SK_DPD").alias("cc_max_dpd_prev"),
            F.avg("AMT_BALANCE").alias("cc_balance_mean_prev"),
            F.max("AMT_BALANCE").alias("cc_balance_max_prev"),
            F.stddev_pop("AMT_BALANCE").alias("cc_balance_vol_prev"),
            F.max("cc_recency").alias("cc_recency_prev"),
        )
    )

    aggregated = (
        per_prev.groupBy("SK_ID_CURR")
        .agg(
            F.count(F.lit(1)).alias("cc_prev_credit_count"),
            F.avg("cc_months_prev").alias("cc_months_prev_mean"),
            F.sum("cc_months_prev").alias("cc_months_total"),
            F.avg("cc_utilization_mean_prev").alias("cc_utilization_mean"),
            F.max("cc_utilization_max_prev").alias("cc_utilization_max"),
            F.avg("cc_payment_to_min_mean_prev").alias("cc_payment_to_min_mean"),
            F.max("cc_payment_to_min_max_prev").alias("cc_payment_to_min_max"),
            F.avg("cc_drawing_to_limit_mean_prev").alias("cc_drawing_to_limit_mean"),
            F.avg("cc_dpd_rate_prev").alias("cc_dpd_rate_mean"),
            F.max("cc_max_dpd_prev").alias("cc_max_dpd"),
            F.avg("cc_balance_mean_prev").alias("cc_balance_mean"),
            F.max("cc_balance_max_prev").alias("cc_balance_max"),
            F.max("cc_balance_vol_prev").alias("cc_balance_vol_max"),
            F.avg("cc_recency_prev").alias("cc_recency_mean"),
        )
    )

    registry = [
        FeatureSpec("cc_prev_credit_count", "credit_card_balance", "SK_ID_CURR", "risk_engineering", "count", "high", True, "Number of previous credit card accounts"),
        FeatureSpec("cc_utilization_mean", "credit_card_balance", "SK_ID_CURR", "risk_engineering", "utilization", "high", True, "Average credit card utilization"),
        FeatureSpec("cc_payment_to_min_mean", "credit_card_balance", "SK_ID_CURR", "risk_engineering", "behavioral", "high", True, "Average payment to minimum due ratio"),
        FeatureSpec("cc_dpd_rate_mean", "credit_card_balance", "SK_ID_CURR", "risk_engineering", "delinquency", "high", True, "Mean delinquency rate for credit cards"),
    ]
    return aggregated, registry