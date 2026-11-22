"""Validate a new customer's details and predict their segment."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH, SEGMENTS_PATH
from src.preprocessing import add_features

# Allowed range for each number the user must provide: (minimum, maximum)
NUMERIC_RULES = {
    "Total_Spend": (1, 10_000_000),
    "Items_Purchased": (1, 10_000),
    "Number_of_Orders": (1, 1_000),
    "Days_Since_Last_Purchase": (0, 3650),
    "Average_Rating": (1, 5),
}


def validate_customer(customer: dict) -> dict:
    """Check the input and return clean numbers. Raise ValueError listing every problem."""
    problems = []
    clean = {}

    for field, (low, high) in NUMERIC_RULES.items():
        try:
            value = float(customer[field])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{field} is missing or not a number")
            continue
        if not low <= value <= high:
            problems.append(f"{field} must be between {low} and {high}")
        clean[field] = value

    discount = str(customer.get("Discount_Applied", "")).strip().lower()
    if discount not in ("yes", "no"):
        problems.append("Discount_Applied must be Yes or No")
    else:
        clean["Discount_Applied"] = 1 if discount == "yes" else 0

    if not problems and clean["Items_Purchased"] < clean["Number_of_Orders"]:
        problems.append("Items_Purchased cannot be less than Number_of_Orders")

    if problems:
        raise ValueError("; ".join(problems))
    return clean


@lru_cache(maxsize=1)
def load_model():
    """Load the saved pipeline once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_segments() -> dict:
    """Load the segment names and strategies."""
    with open(SEGMENTS_PATH, encoding="utf-8") as file:
        return json.load(file)


def predict_segment(customer: dict) -> dict:
    """Return the predicted cluster number plus its name, summary and strategy."""
    clean = validate_customer(customer)
    frame = add_features(pd.DataFrame([clean]))
    cluster = int(load_model().predict(frame[FEATURES])[0])
    return {"cluster": cluster, **load_segments()[str(cluster)]}