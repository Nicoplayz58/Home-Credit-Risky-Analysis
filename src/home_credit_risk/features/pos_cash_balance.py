from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from ..registry import FeatureSpec


def build_pos_cash_features(pos_cash: DataFrame) -> tuple[DataFrame, list[FeatureSpec]]:
    pc = pos_cash.select("SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF")
    pc = pc.withColumn("has_dpd", F.when(F.col("SK_DPD") > 0, F.lit(1)).otherwise(F.lit(0)))
    pc = pc.withColumn("has_dpd_def", F.when(F.col("SK_DPD_DEF") > 0, F.lit(1)).otherwise(F.lit(0)))
    pc = pc.withColumn("instalment_gap", F.col("CNT_INSTALMENT") - F.col("CNT_INSTALMENT_FUTURE"))
    pc = pc.withColumn("remaining_ratio", F.when((F.col("CNT_INSTALMENT") == 0) | F.col("CNT_INSTALMENT").isNull(), F.lit(None)).otherwise(F.col("CNT_INSTALMENT_FUTURE") / F.col("CNT_INSTALMENT")))
    pc = pc.withColumn("activity_recency", F.abs(F.col("MONTHS_BALANCE")))

    per_prev = (
        pc.groupBy("SK_ID_CURR", "SK_ID_PREV")
        .agg(
            F.count(F.lit(1)).alias("pc_months_prev"),
            F.avg("has_dpd").alias("pc_dpd_rate_prev"),
            F.avg("has_dpd_def").alias("pc_dpd_def_rate_prev"),
            F.max("SK_DPD").alias("pc_max_dpd_prev"),
            F.avg("SK_DPD").alias("pc_avg_dpd_prev"),
            F.avg("remaining_ratio").alias("pc_remaining_ratio_prev"),
            F.avg("instalment_gap").alias("pc_instalment_gap_prev"),
            F.max("activity_recency").alias("pc_recency_prev"),
        )
    )

    aggregated = (
        per_prev.groupBy("SK_ID_CURR")
        .agg(
            F.count(F.lit(1)).alias("pc_prev_credit_count"),
            F.avg("pc_months_prev").alias("pc_months_prev_mean"),
            F.sum("pc_months_prev").alias("pc_months_total"),
            F.avg("pc_dpd_rate_prev").alias("pc_dpd_rate_mean"),
            F.max("pc_dpd_rate_prev").alias("pc_dpd_rate_max"),
            F.avg("pc_dpd_def_rate_prev").alias("pc_dpd_def_rate_mean"),
            F.max("pc_max_dpd_prev").alias("pc_max_dpd"),
            F.avg("pc_avg_dpd_prev").alias("pc_avg_dpd"),
            F.avg("pc_remaining_ratio_prev").alias("pc_remaining_ratio_mean"),
            F.avg("pc_instalment_gap_prev").alias("pc_instalment_gap_mean"),
            F.avg("pc_recency_prev").alias("pc_recency_mean"),
        )
    )

    registry = [
        FeatureSpec("pc_prev_credit_count", "POS_CASH_balance", "SK_ID_CURR", "risk_engineering", "count", "high", True, "Number of previous POS/cash credits"),
        FeatureSpec("pc_dpd_rate_mean", "POS_CASH_balance", "SK_ID_CURR", "risk_engineering", "delinquency", "high", True, "Mean delinquency rate across POS/cash credits"),
        FeatureSpec("pc_remaining_ratio_mean", "POS_CASH_balance", "SK_ID_CURR", "risk_engineering", "utilization", "high", True, "Remaining installment ratio across POS/cash credits"),
    ]
    return aggregated, registry