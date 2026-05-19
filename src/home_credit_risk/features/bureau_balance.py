from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F

from ..registry import FeatureSpec


def build_bureau_balance_features(bureau_balance: DataFrame) -> tuple[DataFrame, list[FeatureSpec]]:
    bb = bureau_balance.select("SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS")
    delinquent_statuses = ["1", "2", "3", "4", "5"]

    bb = bb.withColumn("is_delinquent", F.when(F.col("STATUS").isin(delinquent_statuses), F.lit(1)).otherwise(F.lit(0)))
    bb = bb.withColumn("is_closed", F.when(F.col("STATUS") == F.lit("C"), F.lit(1)).otherwise(F.lit(0)))
    bb = bb.withColumn("is_unknown", F.when(F.col("STATUS") == F.lit("X"), F.lit(1)).otherwise(F.lit(0)))

    last_status = (
        bb.withColumn("rn", F.row_number().over(Window.partitionBy("SK_ID_BUREAU").orderBy(F.col("MONTHS_BALANCE").desc())))
        .filter(F.col("rn") == 1)
        .select("SK_ID_BUREAU", F.col("STATUS").alias("bb_last_status"))
    )

    trend = (
        bb.groupBy("SK_ID_BUREAU")
        .agg(
            F.min("MONTHS_BALANCE").alias("bb_months_min"),
            F.max("MONTHS_BALANCE").alias("bb_months_max"),
            F.count(F.lit(1)).alias("bb_months_observed"),
            F.sum("is_delinquent").alias("bb_delinquent_months"),
            F.sum("is_closed").alias("bb_closed_months"),
            F.sum("is_unknown").alias("bb_unknown_months"),
            F.avg("is_delinquent").alias("bb_delinquency_rate"),
            F.avg("is_closed").alias("bb_closed_rate"),
            F.avg("is_unknown").alias("bb_unknown_rate"),
            F.stddev_pop("is_delinquent").alias("bb_delinquency_volatility"),
            F.max(F.when(F.col("is_delinquent") == 1, F.col("MONTHS_BALANCE"))).alias("bb_last_delinquency_month"),
        )
        .withColumn("bb_recency_months", F.abs(F.col("bb_months_max")))
        .withColumn("bb_delinquency_span", F.col("bb_months_max") - F.col("bb_months_min"))
    )

    result = trend.join(last_status, on="SK_ID_BUREAU", how="left")

    registry = [
        FeatureSpec("bb_months_observed", "bureau_balance", "SK_ID_BUREAU", "risk_engineering", "count", "medium", True, "Number of observed bureau balance months"),
        FeatureSpec("bb_delinquency_rate", "bureau_balance", "SK_ID_BUREAU", "risk_engineering", "rate", "high", True, "Share of delinquent months in bureau balance history"),
        FeatureSpec("bb_recency_months", "bureau_balance", "SK_ID_BUREAU", "risk_engineering", "recency", "medium", True, "Recency of latest bureau balance observation"),
        FeatureSpec("bb_last_status", "bureau_balance", "SK_ID_BUREAU", "risk_engineering", "categorical", "medium", True, "Most recent bureau balance status"),
    ]
    return result, registry