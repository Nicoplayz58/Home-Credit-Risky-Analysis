from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from .schemas import build_schema


def read_csv_table(spark: SparkSession, path: Path, delimiter: str = ",") -> DataFrame:
    import csv

    with path.open("r", encoding="utf-8") as handle:
        columns = next(csv.reader(handle, delimiter=delimiter))
    schema = build_schema(columns)
    return (
        spark.read.option("header", True)
        .option("delimiter", delimiter)
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(str(path))
    )


def write_parquet(df: DataFrame, path: Path, mode: str = "overwrite", partition_by: list[str] | None = None) -> None:
    writer = df.write.mode(mode).format("parquet")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(str(path))