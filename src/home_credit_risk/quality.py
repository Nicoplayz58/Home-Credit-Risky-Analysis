from __future__ import annotations

from pyspark.sql import DataFrame, functions as F


def basic_profile(df: DataFrame) -> DataFrame:
    total_rows = df.count()
    profile_rows = []

    for column in df.columns:
        metrics = df.agg(
            (F.count(F.when(F.col(column).isNull(), 1)) / F.lit(total_rows) * 100).alias("null_pct"),
            F.approx_count_distinct(F.col(column)).alias("approx_cardinality"),
            F.lit(str(df.schema[column].dataType)).alias("data_type"),
        ).withColumn("feature", F.lit(column))
        profile_rows.append(metrics)

    if not profile_rows:
        return df.sparkSession.createDataFrame([], "feature string, null_pct double, approx_cardinality bigint, data_type string")

    result = profile_rows[0]
    for profile_df in profile_rows[1:]:
        result = result.unionByName(profile_df)
    return result.select("feature", "null_pct", "approx_cardinality", "data_type")