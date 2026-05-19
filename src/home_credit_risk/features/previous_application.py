from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from ..registry import FeatureSpec


_LEAKY_COLS = {
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
}


def build_previous_application_features(previous_application: DataFrame) -> tuple[DataFrame, list[FeatureSpec]]:
    pa = previous_application.drop(*[c for c in _LEAKY_COLS if c in previous_application.columns])

    pa = pa.withColumn("is_approved", F.when(F.col("NAME_CONTRACT_STATUS") == F.lit("Approved"), F.lit(1)).otherwise(F.lit(0)))
    pa = pa.withColumn("is_refused", F.when(F.col("NAME_CONTRACT_STATUS") == F.lit("Refused"), F.lit(1)).otherwise(F.lit(0)))
    pa = pa.withColumn("is_cancelled", F.when(F.col("NAME_CONTRACT_STATUS") == F.lit("Canceled"), F.lit(1)).otherwise(F.lit(0)))
    pa = pa.withColumn("prev_recency_days", F.abs(F.col("DAYS_DECISION")))
    pa = pa.withColumn("application_credit_ratio", F.when((F.col("AMT_APPLICATION") == 0) | F.col("AMT_APPLICATION").isNull(), F.lit(None)).otherwise(F.col("AMT_CREDIT") / F.col("AMT_APPLICATION")))
    pa = pa.withColumn("approval_gap", F.col("AMT_CREDIT") - F.col("AMT_APPLICATION"))
    pa = pa.withColumn("down_payment_ratio", F.when((F.col("AMT_APPLICATION") == 0) | F.col("AMT_APPLICATION").isNull(), F.lit(None)).otherwise(F.col("AMT_DOWN_PAYMENT") / F.col("AMT_APPLICATION")))
    pa = pa.withColumn("contract_completion_ratio", F.when((F.col("CNT_PAYMENT") == 0) | F.col("CNT_PAYMENT").isNull(), F.lit(None)).otherwise(F.col("AMT_ANNUITY") / F.col("CNT_PAYMENT")))

    aggregated = (
        pa.groupBy("SK_ID_CURR")
        .agg(
            F.count(F.lit(1)).alias("prev_app_count"),
            F.sum("is_approved").alias("prev_app_approved_count"),
            F.sum("is_refused").alias("prev_app_refused_count"),
            F.sum("is_cancelled").alias("prev_app_cancelled_count"),
            F.avg("is_approved").alias("prev_app_approval_rate"),
            F.avg("prev_recency_days").alias("prev_app_recency_days_mean"),
            F.min("prev_recency_days").alias("prev_app_recency_days_min"),
            F.max("prev_recency_days").alias("prev_app_recency_days_max"),
            F.avg("application_credit_ratio").alias("prev_app_credit_ratio_mean"),
            F.max("application_credit_ratio").alias("prev_app_credit_ratio_max"),
            F.avg("down_payment_ratio").alias("prev_app_down_payment_ratio_mean"),
            F.avg("approval_gap").alias("prev_app_approval_gap_mean"),
            F.avg("CNT_PAYMENT").alias("prev_app_cnt_payment_mean"),
            F.avg("NFLAG_INSURED_ON_APPROVAL").alias("prev_app_insured_rate"),
            F.countDistinct("NAME_CONTRACT_TYPE").alias("prev_app_contract_type_diversity"),
            F.countDistinct("NAME_CLIENT_TYPE").alias("prev_app_client_type_diversity"),
            F.countDistinct("PRODUCT_COMBINATION").alias("prev_app_product_combination_diversity"),
        )
    )

    registry = [
        FeatureSpec("prev_app_count", "previous_application", "SK_ID_CURR", "risk_engineering", "count", "medium", True, "Number of historical previous applications"),
        FeatureSpec("prev_app_approval_rate", "previous_application", "SK_ID_CURR", "risk_engineering", "rate", "high", True, "Historical approval rate"),
        FeatureSpec("prev_app_recency_days_mean", "previous_application", "SK_ID_CURR", "risk_engineering", "recency", "medium", True, "Average recency of previous applications"),
        FeatureSpec("prev_app_credit_ratio_mean", "previous_application", "SK_ID_CURR", "risk_engineering", "ratio", "high", True, "Ratio of credit to application amount in previous requests"),
    ]
    return aggregated, registry