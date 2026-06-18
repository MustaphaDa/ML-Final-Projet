from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project settings
CITY = "Malaga"
RANDOM_SEED = 42
TARGET_COLUMN = "price"
TEST_SIZE = 0.2

# Paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Raw file names (place downloads in data/raw/)
LISTINGS_FILE = "listings.csv"
REVIEWS_FILE = "reviews.csv"
CALENDAR_FILE = "calendar.csv.gz"
OUTPUT_FILE = "malaga_modeling_table.csv"

# Cleaning bounds for Malaga
PRICE_LOWER_QUANTILE = 0.01
PRICE_UPPER_QUANTILE = 0.99
LATITUDE_MIN = 36.60
LATITUDE_MAX = 36.90
LONGITUDE_MIN = -4.60
LONGITUDE_MAX = -4.20
