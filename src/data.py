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
]

TEXT_COLUMNS = ["name", "description", "neighborhood_overview", "host_about"]
PERCENTAGE_COLUMNS = ["host_response_rate", "host_acceptance_rate"]
BOOLEAN_COLUMNS = ["host_is_superhost", "host_identity_verified", "instant_bookable", "has_availability"]


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
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r"[\$€,]", "", regex=True).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def clean_percentage(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def clean_boolean(series: pd.Series) -> pd.Series:
    mapping = {"t": 1, "f": 0, "true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
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
    mask = (
        df["latitude"].between(LATITUDE_MIN, LATITUDE_MAX)
        & df["longitude"].between(LONGITUDE_MIN, LONGITUDE_MAX)
    )
    return df.loc[mask].copy()


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    agg_spec: dict = {"review_count": ("date", "count")}
    if "comments" in reviews.columns:
        agg_spec["review_text"] = ("comments", lambda s: " ".join(s.dropna().astype(str)))

    grouped = reviews.groupby("listing_id").agg(**agg_spec).reset_index()
    date_stats = (
        reviews.groupby("listing_id")["date"]
        .agg(first_review_date="min", last_review_date="max")
        .reset_index()
    )
    grouped = grouped.merge(date_stats, on="listing_id", how="left")
    grouped["review_span_days"] = (grouped["last_review_date"] - grouped["first_review_date"]).dt.days

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
    if "amenities" in cleaned.columns:
        cleaned["amenities_count"] = cleaned["amenities"].apply(count_amenities)
    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(strip_html).replace("", np.nan)

    drop_cols = [col for col in LEAKY_COLUMNS + DROP_COLUMNS if col in cleaned.columns]
    return cleaned.drop(columns=drop_cols).reset_index(drop=True)


def build_modeling_table(listings: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    listings_clean = clean_listings(listings)
    reviews_agg = aggregate_reviews(reviews)
    df = listings_clean.merge(reviews_agg, left_on="id", right_on="listing_id", how="left")
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    return df.reset_index(drop=True)


def save_modeling_table() -> Path:
    """Load raw data, clean, merge, and write the processed CSV."""
    df = build_modeling_table(load_listings(), load_reviews())
    output_path = PROCESSED_DATA_DIR / OUTPUT_FILE
    df.to_csv(output_path, index=False)
    return output_path
