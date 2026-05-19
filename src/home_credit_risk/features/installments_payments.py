from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from ..registry import FeatureSpec


def build_installments_features(installments: DataFrame) -> tuple[DataFrame, list[FeatureSpec]]:
    ip = installments.select("SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT")
    ip = ip.withColumn("payment_delay_days", F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT"))
    ip = ip.withColumn("paid_on_time", F.when(F.col("payment_delay_days") <= 0, F.lit(1)).otherwise(F.lit(0)))
    ip = ip.withColumn("late_payment", F.when(F.col("payment_delay_days") > 0, F.lit(1)).otherwise(F.lit(0)))
    ip = ip.withColumn("underpayment_amount", F.greatest(F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT"), F.lit(0.0)))
    ip = ip.withColumn("payment_ratio", F.when((F.col("AMT_INSTALMENT") == 0) | F.col("AMT_INSTALMENT").isNull(), F.lit(None)).otherwise(F.col("AMT_PAYMENT") / F.col("AMT_INSTALMENT")))
    ip = ip.withColumn("instalment_recency_days", F.abs(F.col("DAYS_INSTALMENT")))

    prev_level = (
        ip.groupBy("SK_ID_CURR", "SK_ID_PREV")
        .agg(
            F.count(F.lit(1)).alias("ip_rows_per_prev"),
            F.avg("payment_delay_days").alias("ip_delay_mean_prev"),
            F.max("payment_delay_days").alias("ip_delay_max_prev"),
            F.avg("paid_on_time").alias("ip_on_time_rate_prev"),
            F.avg("late_payment").alias("ip_late_rate_prev"),
            F.avg("payment_ratio").alias("ip_payment_ratio_mean_prev"),
            F.stddev_pop("payment_ratio").alias("ip_payment_ratio_vol_prev"),
            F.sum("underpayment_amount").alias("ip_underpayment_total_prev"),
            F.max("instalment_recency_days").alias("ip_recency_days_prev"),
        )
    )

    aggregated = (
        prev_level.groupBy("SK_ID_CURR")
        .agg(
            F.count(F.lit(1)).alias("ip_prev_credit_count"),
            F.avg("ip_rows_per_prev").alias("ip_rows_per_prev_mean"),
            F.sum("ip_rows_per_prev").alias("ip_rows_total"),
            F.avg("ip_delay_mean_prev").alias("ip_delay_mean"),
            F.max("ip_delay_max_prev").alias("ip_delay_max"),
            F.avg("ip_on_time_rate_prev").alias("ip_on_time_rate"),
            F.avg("ip_late_rate_prev").alias("ip_late_rate"),
            F.avg("ip_payment_ratio_mean_prev").alias("ip_payment_ratio_mean"),
            F.max("ip_payment_ratio_vol_prev").alias("ip_payment_ratio_vol_max"),
            F.sum("ip_underpayment_total_prev").alias("ip_underpayment_total"),
            F.avg("ip_recency_days_prev").alias("ip_recency_days_mean"),
            F.max("ip_recency_days_prev").alias("ip_recency_days_max"),
        )
    )

    registry = [
        FeatureSpec("ip_prev_credit_count", "installments_payments", "SK_ID_CURR", "risk_engineering", "count", "high", True, "Number of previous installment-linked credits"),
        FeatureSpec("ip_on_time_rate", "installments_payments", "SK_ID_CURR", "risk_engineering", "behavioral", "high", True, "Average on-time payment rate across previous loans"),
        FeatureSpec("ip_delay_mean", "installments_payments", "SK_ID_CURR", "risk_engineering", "behavioral", "high", True, "Average payment delay across previous loans"),
        FeatureSpec("ip_payment_ratio_mean", "installments_payments", "SK_ID_CURR", "risk_engineering", "ratio", "high", True, "Average payment-to-installment ratio"),
    ]
    return aggregated, registry