from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, functions as F


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    details: str


def count_duplicates(df: DataFrame, keys: list[str]) -> int:
    return df.groupBy(*keys).count().filter(F.col("count") > 1).count()


def assert_unique_keys(df: DataFrame, keys: list[str], dataset_name: str) -> ValidationResult:
    duplicates = count_duplicates(df, keys)
    if duplicates > 0:
        return ValidationResult(dataset_name, False, f"{duplicates} duplicated key groups found on {keys}")
    return ValidationResult(dataset_name, True, f"unique keys validated on {keys}")


def assert_no_many_to_many(left: DataFrame, right: DataFrame, join_keys: list[str]) -> ValidationResult:
    right_dup = count_duplicates(right, join_keys)
    left_dup = count_duplicates(left, join_keys)
    passed = right_dup == 0 and left_dup == 0
    details = f"left_duplicates={left_dup}, right_duplicates={right_dup}, keys={join_keys}"
    return ValidationResult("join_cardinality", passed, details)


def safe_left_join(left: DataFrame, right: DataFrame, join_keys: list[str], right_name: str) -> DataFrame:
    join_check = assert_no_many_to_many(left, right, join_keys)
    if not join_check.passed:
        raise ValueError(f"Unsafe join against {right_name}: {join_check.details}")
    return left.join(right, on=join_keys, how="left")