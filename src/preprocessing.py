"""Cleaning, feature engineering and the scikit-learn preprocessing pipeline."""
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (GENDER_MAP, LOG_FEATURES, MAX_AGE, MIN_AGE, PLAIN_FEATURES)

logger = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, columns: list) -> None:
    """Raise a clear error if any expected column is missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw customer table (training use)."""
    _require_columns(df, ["Gender", "City", "Age", "Discount_Applied"])
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate rows", before - len(df))

    df["Gender"] = df["Gender"].str.strip().str.lower().map(GENDER_MAP)
    df["City"] = df["City"].str.strip().str.title()

    bad_age = df["Age"].notna() & ~df["Age"].between(MIN_AGE, MAX_AGE)
    df.loc[bad_age, "Age"] = np.nan
    logger.info("Marked %d impossible ages as missing", bad_age.sum())

    df["Discount_Applied"] = (
        df["Discount_Applied"].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0})
    )
    if df["Discount_Applied"].isna().any():
        raise ValueError("Discount_Applied contains values other than Yes/No")
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Avg_Order_Value and Items_Per_Order. Used in training AND by the app."""
    _require_columns(df, ["Total_Spend", "Items_Purchased", "Number_of_Orders"])
    if (df["Number_of_Orders"] <= 0).any():
        raise ValueError("Number_of_Orders must be at least 1")
    df = df.copy()
    df["Avg_Order_Value"] = df["Total_Spend"] / df["Number_of_Orders"]
    df["Items_Per_Order"] = df["Items_Purchased"] / df["Number_of_Orders"]
    return df


def build_preprocessor() -> ColumnTransformer:
    """Log + scale the skewed features; impute + scale the rest."""
    log_part = Pipeline([("log", FunctionTransformer(np.log1p)),
                         ("scale", StandardScaler())])
    plain_part = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())])
    return ColumnTransformer([
        ("log_features", log_part, LOG_FEATURES),
        ("plain_features", plain_part, PLAIN_FEATURES),
    ])