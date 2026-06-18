"""Build and save the cleaned modeling table."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import CITY, TARGET_COLUMN
from src.data import load_listings, load_reviews, save_modeling_table


def main() -> None:
    print(f"Loading data for {CITY}...")

    listings = load_listings()
    reviews = load_reviews()
    print(f"  Raw listings: {listings.shape[0]:,} rows")
    print(f"  Raw reviews:  {reviews.shape[0]:,} rows")

    output_path = save_modeling_table()
    print(f"Saved -> {output_path}")

    df = pd.read_csv(output_path)
    price = df[TARGET_COLUMN]
    print("\nPrice summary (EUR):")
    print(f"  min:    {price.min():.2f}")
    print(f"  median: {price.median():.2f}")
    print(f"  mean:   {price.mean():.2f}")
    print(f"  max:    {price.max():.2f}")


if __name__ == "__main__":
    main()
