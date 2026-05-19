from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    source_table: str
    granularity: str
    owner: str
    feature_type: str
    risk_potential: str
    temporal_safety: bool
    business_description: str


def registry_to_dataframe(spark: SparkSession, specs: Iterable[FeatureSpec]):
    rows = [asdict(spec) for spec in specs]
    return spark.createDataFrame(rows)


def write_registry(df: DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").option("header", True).csv(str(output_path / "csv"))
    df.write.mode("overwrite").parquet(str(output_path / "parquet"))