"""
BAN-501 Final Project: Data Loading Example

This script demonstrates how to load and inspect the publication dataset.
"""

import polars as pl

# Load the dataset
df = pl.read_parquet("data/ai_publications_2020_plus.parquet")

# Basic info
print(f"Total records: {len(df):,}")
print(f"Columns: {df.columns}")
print()

# Year distribution
print("Records by year:")
print(df.group_by("year").len().sort("year"))
print()

# Publication type distribution
print("Records by publication type:")
print(df.group_by("publication_type").len().sort("len", descending=True))
print()

# Sample records
print("Sample abstracts:")
print("-" * 80)
for row in df.head(3).iter_rows(named=True):
    print(f"Title: {row['title'][:80]}...")
    print(f"Year: {row['year']} | Type: {row['publication_type']}")
    print(f"Abstract: {row['abstract'][:200]}...")
    print("-" * 80)
