"""Train the CustomerIQ model and save all artifacts.

Run from the project root:   python -m src.train
"""
import logging

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.config import (CLEAN_DATA_PATH, CLUSTER_PROFILES_PATH, FEATURES,
                        MODEL_PATH, N_CLUSTERS, RANDOM_STATE, RAW_DATA_PATH)
from src.preprocessing import add_features, build_preprocessor, clean_data

logger = logging.getLogger(__name__)

PROFILE_COLUMNS = ["Total_Spend", "Number_of_Orders", "Avg_Order_Value",
                   "Days_Since_Last_Purchase", "Average_Rating",
                   "Discount_Applied", "Items_Per_Order"]


def train() -> Pipeline:
    """Clean the raw data, fit the pipeline, and save the model and profiles."""
    raw = pd.read_csv(RAW_DATA_PATH)
    df = add_features(clean_data(raw))

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=10,
                          max_iter=300, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(df[FEATURES])
    df["Cluster"] = pipeline.predict(df[FEATURES])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    profile = df.groupby("Cluster")[PROFILE_COLUMNS].mean().round(2)
    profile.to_csv(CLUSTER_PROFILES_PATH)

    logger.info("Inertia: %d", round(pipeline.named_steps["kmeans"].inertia_))
    logger.info("Cluster sizes: %s", df["Cluster"].value_counts().sort_index().tolist())
    logger.info("Saved model to %s", MODEL_PATH)
    return pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train()