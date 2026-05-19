from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession

from .config import ProjectConfig


def build_spark_session(config: ProjectConfig) -> SparkSession:
    spark_cfg = config.raw["spark"]
    project_root = config.root
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        os.environ.setdefault("PYSPARK_PYTHON", str(venv_python))
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", str(venv_python))

    builder = (
        SparkSession.builder.appName(spark_cfg["app_name"])
        .master(spark_cfg["master"])
        .config("spark.sql.shuffle.partitions", str(spark_cfg["shuffle_partitions"]))
        .config("spark.sql.adaptive.enabled", str(spark_cfg["adaptive_enabled"]).lower())
        .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", str(spark_cfg["dynamic_partition_pruning"]).lower())
        .config("spark.sql.broadcastTimeout", str(spark_cfg["broadcast_timeout"]))
        .config("spark.sql.parquet.compression.codec", config.get("output", "compression", default="zstd"))
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.executor.memory", spark_cfg["executor_memory"])
        .config("spark.sql.files.maxRecordsPerFile", str(config.get("output", "max_records_per_file", default=500000)))
        .config("spark.hadoop.io.native.lib", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
    )
    return builder.getOrCreate()