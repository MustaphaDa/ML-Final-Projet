"""Load raw Airbnb files and build the modeling table."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CALENDAR_FILE,
    LATITUDE_MAX,
    LATITUDE_MIN,
    LISTINGS_FILE,
    LONGITUDE_MAX,
    LONGITUDE_MIN,
    OUTPUT_FILE,
    PRICE_LOWER_QUANTILE,
    PRICE_UPPER_QUANTILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REVIEWS_FILE,
    TARGET_COLUMN,
)

LEAKY_COLUMNS = [
    "estimated_revenue_l365d",
    "estimated_occupancy_l365d",
]

DROP_COLUMNS = [
    "listing_url",
    "scrape_id",
    "last_scraped",
    "source",
    "picture_url",
    "host_url",
    "host_thumbnail_url",
    "host_picture_url",
    "calendar_updated",
    "calendar_last_scraped",
    "neighbourhood",
    "host_verifications",
    "host_has_profile_pic",
    "host_neighbourhood",  # ~87% missing; neighbourhood_cleansed is enough
]

TEXT_COLUMNS = ["name", "description", "neighborhood_overview", "host_about"]
PERCENTAGE_COLUMNS = ["host_response_rate", "host_acceptance_rate"]
BOOLEAN_COLUMNS = [
    "host_is_superhost",
    "host_identity_verified",
    "instant_bookable",
    "has_availability",
]
DATE_COLUMNS = ["host_since", "first_review", "last_review"]
REVIEW_SCORE_COLUMNS = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def load_listings() -> pd.DataFrame:
    return _read_csv(RAW_DATA_DIR / LISTINGS_FILE)


def load_reviews() -> pd.DataFrame:
    return _read_csv(RAW_DATA_DIR / REVIEWS_FILE)


def load_calendar() -> pd.DataFrame | None:
    """Return calendar data, or None if the file has not been downloaded yet."""
    path = RAW_DATA_DIR / CALENDAR_FILE
    if not path.exists():
        return None
    return _read_csv(path)


def clean_price(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[\$€,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def clean_percentage(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def clean_boolean(series: pd.Series) -> pd.Series:
    mapping = {
        "t": 1,
        "f": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
    }
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(mapping)


def strip_html(text: str) -> str:
    if pd.isna(text):
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", without_tags).strip()


def count_amenities(value) -> int:
    if pd.isna(value):
        return 0
    return len(re.findall(r'"[^"]+"', str(value)))


def parse_bathrooms_text(series: pd.Series) -> pd.Series:
    """Extract numeric bathroom count from strings like '1 bath' or '1.5 shared baths'."""
    extracted = series.astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def convert_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def drop_bad_columns(df: pd.DataFrame, missing_threshold: float = 0.95) -> pd.DataFrame:
    """Drop empty, constant, leaky, and very sparse columns."""
    drop = set(DROP_COLUMNS + LEAKY_COLUMNS)

    for col in df.columns:
        missing_rate = df[col].isna().mean()
        if missing_rate >= missing_threshold:
            drop.add(col)
        elif df[col].notna().any() and df[col].nunique(dropna=True) <= 1:
            drop.add(col)

    existing = [col for col in drop if col in df.columns]
    return df.drop(columns=existing)


def cleaning_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize column types and missing values after cleaning."""
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_pct": round(df[col].isna().mean() * 100, 1),
                "unique": df[col].nunique(dropna=True),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def remove_price_outliers(
    df: pd.DataFrame,
    price_col: str = TARGET_COLUMN,
    lower_quantile: float = PRICE_LOWER_QUANTILE,
    upper_quantile: float = PRICE_UPPER_QUANTILE,
) -> pd.DataFrame:
    prices = clean_price(df[price_col])
    valid = prices.notna() & (prices > 0)
    lower = prices[valid].quantile(lower_quantile)
    upper = prices[valid].quantile(upper_quantile)
    mask = valid & prices.between(lower, upper)
    result = df.loc[mask].copy()
    result[price_col] = prices.loc[mask]
    return result


def filter_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["latitude"].between(LATITUDE_MIN, LATITUDE_MAX) & df["longitude"].between(
        LONGITUDE_MIN, LONGITUDE_MAX
    )
    return df.loc[mask].copy()


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    agg_spec: dict = {"review_count": ("date", "count")}
    if "comments" in reviews.columns:
        agg_spec["review_text"] = (
            "comments",
            lambda s: " ".join(s.dropna().astype(str)),
        )

    grouped = reviews.groupby("listing_id").agg(**agg_spec).reset_index()
    date_stats = (
        reviews.groupby("listing_id")["date"]
        .agg(first_review_date="min", last_review_date="max")
        .reset_index()
    )
    grouped = grouped.merge(date_stats, on="listing_id", how="left")
    grouped["review_span_days"] = (
        grouped["last_review_date"] - grouped["first_review_date"]
    ).dt.days

    if "review_text" in grouped.columns:
        grouped["review_text_length"] = grouped["review_text"].str.len()

    return grouped


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned[TARGET_COLUMN] = clean_price(cleaned[TARGET_COLUMN])
    cleaned = cleaned[cleaned[TARGET_COLUMN].notna() & (cleaned[TARGET_COLUMN] > 0)]
    cleaned = remove_price_outliers(cleaned)
    cleaned = filter_coordinates(cleaned)

    for col in PERCENTAGE_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = clean_percentage(cleaned[col])
    for col in BOOLEAN_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = clean_boolean(cleaned[col])
    cleaned = convert_dates(cleaned, DATE_COLUMNS)

    if "bathrooms_text" in cleaned.columns:
        cleaned["bathrooms_count"] = parse_bathrooms_text(cleaned["bathrooms_text"])
    if "amenities" in cleaned.columns:
        cleaned["amenities_count"] = cleaned["amenities"].apply(count_amenities)
    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(strip_html).replace("", np.nan)

    cleaned = drop_bad_columns(cleaned)
    return cleaned.reset_index(drop=True)


def build_modeling_table(listings: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    listings_clean = clean_listings(listings)
    reviews_agg = aggregate_reviews(reviews)
    df = listings_clean.merge(
        reviews_agg, left_on="id", right_on="listing_id", how="left"
    )
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["has_reviews"] = (df["review_count"] > 0).astype(int)

    # Drop redundant merge key
    if "listing_id" in df.columns:
        df = df.drop(columns=["listing_id"])

    return df.reset_index(drop=True)


# =====================================================================
# === SPATIAL FEATURE ENGINEERING                                   ===
# =====================================================================

# Malaga geographic coordinates and landmarks
MALAGA_CENTER = (36.721071745393665, -4.422086495998684)  # Plaza de la Constitución

# Extracted coordinates from the Malaga beach coastline map (latitude, longitude)
BEACH_POINTS = np.array(
    [
        [36.692789153127954, -4.440228665971509],  # Misericordia Beach (West)
        [36.70241003211757, -4.433250476269645],  # Huelin Beach
        [36.717253695523326, -4.410496252052286],  # Malagueta Beach (1)
        [36.72005173741972, -4.402729784919005],  # Malagueta Beach (2)
        [36.722703020692414, -4.392852249502834],  # Malagueta Beach (3)
        [36.72141890197016, -4.38255262299423],  # Pedregalejo Beach
        [36.71866654638932, -4.360032646495201],  # El Palo Beach (East)
    ]
)


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance to the historic center and the closest beach point using Haversine formula."""
    R = 6371.0  # Earth's radius in kilometers

    # 1. DISTANCE TO HISTORIC CENTER
    lat1, lon1 = np.radians(MALAGA_CENTER[0]), np.radians(MALAGA_CENTER[1])
    lat2, lon2 = np.radians(df["latitude"]), np.radians(df["longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance_to_center"] = R * c

    # 2. DISTANCE TO NEAREST BEACH POINT
    def calculate_minimum_beach_distance(row):
        property_lat, property_lon = np.radians(row["latitude"]), np.radians(
            row["longitude"]
        )
        beach_lat = np.radians(BEACH_POINTS[:, 0])
        beach_lon = np.radians(BEACH_POINTS[:, 1])

        dlat_p = beach_lat - property_lat
        dlon_p = beach_lon - property_lon
        a_p = (
            np.sin(dlat_p / 2) ** 2
            + np.cos(property_lat) * np.cos(beach_lat) * np.sin(dlon_p / 2) ** 2
        )
        c_p = 2 * np.arctan2(np.sqrt(a_p), np.sqrt(1 - a_p))
        return np.min(R * c_p)

    df["distance_to_beach"] = df.apply(calculate_minimum_beach_distance, axis=1)

    return df


# =====================================================================


def save_modeling_table() -> Path:
    """Load raw data, clean, merge, and write the processed CSV."""
    df = build_modeling_table(load_listings(), load_reviews())

    # Apply the spatial feature engineering process
    df = add_spatial_features(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / OUTPUT_FILE
    fallback_path = PROCESSED_DATA_DIR / "malaga_modeling_table_latest.csv"

    for path in (output_path, fallback_path):
        try:
            df.to_csv(path, index=False)
            if path != output_path:
                print(
                    f"Could not overwrite {output_path.name} (file may be open in Excel or the editor). "
                    f"Saved to {path.name} instead."
                )
            return path
        except PermissionError:
            continue

    raise PermissionError(
        f"Cannot write processed data. Close {output_path.name} if it is open, then run again."
    )
