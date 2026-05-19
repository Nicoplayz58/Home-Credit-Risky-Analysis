from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType


_STRING_PREFIXES = (
    "NAME_",
    "CODE_",
    "OCCUPATION_",
    "ORGANIZATION_",
    "WEEKDAY_",
    "HOUSETYPE_",
    "EMERGENCYSTATE_",
    "PRODUCT_",
    "CHANNEL_",
    "CREDIT_ACTIVE",
    "CREDIT_CURRENCY",
    "CREDIT_TYPE",
    "NAME_CONTRACT_STATUS",
    "NAME_PAYMENT_TYPE",
    "NAME_CLIENT_TYPE",
    "NAME_GOODS_CATEGORY",
    "NAME_PORTFOLIO",
    "NAME_YIELD_GROUP",
    "NAME_SELLER_INDUSTRY",
    "NAME_CASH_LOAN_PURPOSE",
    "NAME_TYPE_SUITE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "WALLSMATERIAL_MODE",
    "FONDKAPREMONT_MODE",
    "STATUS",
)


def _infer_type(column_name: str):
    name = column_name.upper()
    if name == "TARGET":
        return IntegerType()
    if name.startswith("SK_ID_"):
        return LongType()
    if name.startswith("FLAG_") or name.startswith("NFLAG_"):
        return IntegerType()
    if name.startswith("CNT_") or name.startswith("NUM_") or name.startswith("OBS_") or name.startswith("DEF_"):
        return IntegerType()
    if name.startswith("DAYS_") or name.startswith("MONTHS_"):
        return IntegerType()
    if name.startswith("AMT_") or name.startswith("RATE_") or name.startswith("EXT_SOURCE_"):
        return DoubleType()
    if name.startswith("REGION_POPULATION_RELATIVE") or name.startswith("TOTALAREA_MODE"):
        return DoubleType()
    if name.startswith(_STRING_PREFIXES):
        return StringType()
    if name in {"HOUR_APPR_PROCESS_START", "SELLERPLACE_AREA"}:
        return IntegerType()
    if name.endswith("_AGE"):
        return IntegerType()
    return DoubleType()


def build_schema(columns: Iterable[str]) -> StructType:
    return StructType([StructField(column, _infer_type(column), True) for column in columns])


@lru_cache(maxsize=None)
def schema_from_header(header_path: str) -> StructType:
    from csv import reader

    with Path(header_path).open("r", encoding="utf-8") as handle:
        columns = next(reader(handle))
    return build_schema(columns)