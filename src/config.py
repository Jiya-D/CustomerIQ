"""Project-wide settings for CustomerIQ."""
from pathlib import Path

# The project root is two levels up from this file: src/config.py -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customers_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "customeriq_kmeans_pipeline.joblib"
SEGMENTS_PATH = PROJECT_ROOT / "models" / "segment_profiles.json"
CLUSTER_PROFILES_PATH = PROJECT_ROOT / "models" / "cluster_profiles.csv"

N_CLUSTERS = 4
RANDOM_STATE = 42

# Ages outside this range are treated as data errors
MIN_AGE, MAX_AGE = 18, 100

LOG_FEATURES = ["Number_of_Orders", "Avg_Order_Value", "Days_Since_Last_Purchase"]
PLAIN_FEATURES = ["Items_Per_Order", "Average_Rating", "Discount_Applied"]
FEATURES = LOG_FEATURES + PLAIN_FEATURES

GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"}