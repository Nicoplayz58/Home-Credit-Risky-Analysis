from __future__ import annotations

import math

from pyspark.sql import DataFrame, functions as F

from _common import ROOT
from home_credit_risk.config import load_config
from home_credit_risk.io import read_csv_table
from home_credit_risk.logger import setup_logger
from home_credit_risk.paths import build_paths
from home_credit_risk.spark_session import build_spark_session


def _psi_for_column(reference: DataFrame, target: DataFrame, column: str, buckets: int = 10) -> float:
    quantiles = reference.approxQuantile(column, [i / buckets for i in range(buckets + 1)], 0.001)
    boundaries = sorted(set(quantiles))
    if len(boundaries) < 3:
        return 0.0

    edges = boundaries[1:-1]

    def _bucket_expr(col_name: str):
        expr = F.when(F.col(col_name).isNull(), F.lit("missing"))
        for idx, edge in enumerate(edges):
            expr = expr.when(F.col(col_name) <= F.lit(edge), F.lit(f"b{idx}"))
        return expr.otherwise(F.lit(f"b{len(edges)}"))

    ref_bucketed = reference.select(F.col(column).alias(column)).withColumn("bucket", _bucket_expr(column))
    tgt_bucketed = target.select(F.col(column).alias(column)).withColumn("bucket", _bucket_expr(column))

    ref_total = ref_bucketed.count()
    tgt_total = tgt_bucketed.count()
    if ref_total == 0 or tgt_total == 0:
        return 0.0

    ref_dist = {row[0]: row[1] / ref_total for row in ref_bucketed.groupBy("bucket").count().collect()}
    tgt_dist = {row[0]: row[1] / tgt_total for row in tgt_bucketed.groupBy("bucket").count().collect()}
    score = 0.0
    for bucket in sorted(set(ref_dist) | set(tgt_dist)):
        ref_share = max(ref_dist.get(bucket, 0.0), 1e-6)
        tgt_share = max(tgt_dist.get(bucket, 0.0), 1e-6)
        score += (tgt_share - ref_share) * math.log(tgt_share / ref_share)
    return float(score)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    paths = build_paths(config)
    logger = setup_logger("drift_monitoring", paths.logs)
    spark = build_spark_session(config)

    train = read_csv_table(spark, paths.raw / config.raw["tables"]["application_train"])
    test = read_csv_table(spark, paths.raw / config.raw["tables"]["application_test"])

    numeric_columns = [c for c in train.columns if c.startswith(("AMT_", "DAYS_", "EXT_SOURCE_", "REGION_", "CNT_", "OBS_", "DEF_"))]
    numeric_columns = [c for c in numeric_columns if c in test.columns]

    results = []
    for column in numeric_columns[:50]:
        psi = _psi_for_column(train, test, column, buckets=config.raw["drift"]["psi_buckets"])
        results.append((column, psi))

    drift_df = spark.createDataFrame(results, ["feature", "psi"]) if results else spark.createDataFrame([], "feature string, psi double")
    drift_df.write.mode("overwrite").parquet(str(paths.checks / "application_drift_psi.parquet"))
    logger.info("Saved drift report to %s", paths.checks)


if __name__ == "__main__":
    main()