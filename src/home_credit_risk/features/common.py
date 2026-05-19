from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F


def safe_ratio(numerator, denominator):
    return F.when(denominator.isNull() | (denominator == 0), F.lit(None)).otherwise(numerator / denominator)


def safe_binary_status(column_name: str, positive_values: list[str]):
    return F.when(F.col(column_name).isin(positive_values), F.lit(1)).otherwise(F.lit(0))


def add_row_rank(df: DataFrame, partition_cols: list[str], order_col: str, rank_col: str = "row_rank") -> DataFrame:
    window = Window.partitionBy(*partition_cols).orderBy(F.col(order_col).asc())
    return df.withColumn(rank_col, F.row_number().over(window))


def linear_trend(df: DataFrame, group_cols: list[str], time_col: str, value_col: str, feature_name: str) -> DataFrame:
    stats = (
        df.groupBy(group_cols)
        .agg(
            F.count(F.lit(1)).alias(f"{feature_name}__n"),
            F.avg(F.col(time_col)).alias(f"{feature_name}__mean_t"),
            F.avg(F.col(value_col)).alias(f"{feature_name}__mean_y"),
            F.avg(F.col(time_col) * F.col(value_col)).alias(f"{feature_name}__mean_ty"),
            F.avg(F.col(time_col) * F.col(time_col)).alias(f"{feature_name}__mean_tt"),
        )
    )
    return stats.withColumn(
        feature_name,
        F.when(
            (F.col(f"{feature_name}__mean_tt") - F.col(f"{feature_name}__mean_t") * F.col(f"{feature_name}__mean_t")) == 0,
            F.lit(0.0),
        ).otherwise(
            (F.col(f"{feature_name}__mean_ty") - F.col(f"{feature_name}__mean_t") * F.col(f"{feature_name}__mean_y"))
            / (F.col(f"{feature_name}__mean_tt") - F.col(f"{feature_name}__mean_t") * F.col(f"{feature_name}__mean_t"))
        ),
    ).drop(
        *[
            c
            for c in stats.columns
            if c.endswith("__n") or c.endswith("__mean_t") or c.endswith("__mean_y") or c.endswith("__mean_ty") or c.endswith("__mean_tt")
        ]
    )


def clip_negative(col_name: str):
    return F.when(F.col(col_name).isNull(), F.lit(None)).otherwise(F.greatest(F.col(col_name), F.lit(0)))