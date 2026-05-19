from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from ..registry import FeatureSpec


def build_bureau_features(bureau: DataFrame, bureau_balance_features: DataFrame) -> tuple[DataFrame, list[FeatureSpec]]:
    b = bureau.select(
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "CREDIT_CURRENCY",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_TYPE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
    )

    b = b.join(bureau_balance_features, on="SK_ID_BUREAU", how="left")

    b = b.withColumn("bureau_has_overdue", F.when(F.col("CREDIT_DAY_OVERDUE") > 0, F.lit(1)).otherwise(F.lit(0)))
    b = b.withColumn("bureau_is_active", F.when(F.col("CREDIT_ACTIVE") == F.lit("Active"), F.lit(1)).otherwise(F.lit(0)))
    b = b.withColumn("bureau_credit_utilization", F.when((F.col("AMT_CREDIT_SUM_LIMIT").isNull()) | (F.col("AMT_CREDIT_SUM_LIMIT") == 0), F.lit(None)).otherwise(F.col("AMT_CREDIT_SUM_DEBT") / F.col("AMT_CREDIT_SUM_LIMIT")))
    b = b.withColumn("bureau_debt_ratio", F.when((F.col("AMT_CREDIT_SUM") == 0) | F.col("AMT_CREDIT_SUM").isNull(), F.lit(None)).otherwise(F.col("AMT_CREDIT_SUM_DEBT") / F.col("AMT_CREDIT_SUM")))
    b = b.withColumn("bureau_recency_days", F.abs(F.col("DAYS_CREDIT")))

    aggregated = (
        b.groupBy("SK_ID_CURR")
        .agg(
            F.count(F.lit(1)).alias("bureau_num_loans"),
            F.sum("bureau_is_active").alias("bureau_active_loans"),
            F.avg("bureau_is_active").alias("bureau_active_rate"),
            F.sum("bureau_has_overdue").alias("bureau_loans_with_overdue"),
            F.avg("bureau_has_overdue").alias("bureau_overdue_rate"),
            F.avg("bureau_recency_days").alias("bureau_recency_days_mean"),
            F.max("bureau_recency_days").alias("bureau_recency_days_max"),
            F.avg("AMT_CREDIT_SUM").alias("bureau_credit_sum_mean"),
            F.sum("AMT_CREDIT_SUM").alias("bureau_credit_sum_total"),
            F.avg("AMT_CREDIT_SUM_DEBT").alias("bureau_debt_mean"),
            F.sum("AMT_CREDIT_SUM_DEBT").alias("bureau_debt_total"),
            F.avg("bureau_credit_utilization").alias("bureau_credit_utilization_mean"),
            F.max("bureau_credit_utilization").alias("bureau_credit_utilization_max"),
            F.avg("bureau_debt_ratio").alias("bureau_debt_ratio_mean"),
            F.max("CREDIT_DAY_OVERDUE").alias("bureau_max_overdue_days"),
            F.sum("CNT_CREDIT_PROLONG").alias("bureau_credit_prolong_total"),
            F.avg("bb_delinquency_rate").alias("bureau_bb_delinquency_rate_mean"),
            F.max("bb_delinquency_rate").alias("bureau_bb_delinquency_rate_max"),
            F.avg("bb_recency_months").alias("bureau_bb_recency_months_mean"),
            F.max("bb_delinquency_volatility").alias("bureau_bb_delinquency_volatility_max"),
            F.avg("bb_months_observed").alias("bureau_bb_months_observed_mean"),
            F.sum(F.when(F.col("CREDIT_ACTIVE") == F.lit("Closed"), F.lit(1)).otherwise(F.lit(0))).alias("bureau_closed_loans"),
            F.countDistinct("CREDIT_TYPE").alias("bureau_credit_type_diversity"),
        )
    )

    registry = [
        FeatureSpec("bureau_num_loans", "bureau", "SK_ID_CURR", "risk_engineering", "count", "medium", True, "Count of bureau credit records per client"),
        FeatureSpec("bureau_active_rate", "bureau", "SK_ID_CURR", "risk_engineering", "rate", "high", True, "Share of active bureau credits"),
        FeatureSpec("bureau_overdue_rate", "bureau", "SK_ID_CURR", "risk_engineering", "rate", "high", True, "Share of bureau credits with overdue days"),
        FeatureSpec("bureau_bb_delinquency_rate_mean", "bureau_balance+bureau", "SK_ID_CURR", "risk_engineering", "behavioral", "high", True, "Average delinquency rate from bureau balance history"),
    ]
    return aggregated, registry