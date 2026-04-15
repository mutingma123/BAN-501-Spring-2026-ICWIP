"""
BAN-501 Final Project: Submission Template

This script shows the expected format for the top-25 predictions parquet file.
Replace the placeholder code with your actual model predictions.
"""

import polars as pl

# Load your dataset
df = pl.read_parquet("data/ai_publications_2020_plus.parquet")

# -----------------------------------------------------------------------------
# YOUR CODE HERE: Generate predictions for all abstracts
# -----------------------------------------------------------------------------
# Example placeholder (replace with your actual model inference):
#
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
#
# model = AutoModelForSequenceClassification.from_pretrained("your_trained_model")
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#
# predictions = []
# for abstract in df["abstract"]:
#     inputs = tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         prob = torch.softmax(outputs.logits, dim=1)[0, 1].item()  # Probability of positive class
#     predictions.append(prob)
#
# df = df.with_columns(pl.Series("predicted_probability", predictions))
# -----------------------------------------------------------------------------

# For demonstration only - replace with real predictions
# This just uses random values as a placeholder
import numpy as np

np.random.seed(42)
df = df.with_columns(
    pl.Series(
        name="predicted_probability",
        values=np.random.random(len(df)),
    )
)

# Select top 25 by predicted probability
top_25 = (
    df.select(["scopus_id", "title", "abstract", "predicted_probability"])
    .sort("predicted_probability", descending=True)
    .head(25)
)

# Verify format
print("Top 25 predictions:")
print(f"Columns: {top_25.columns}")
print(f"Shape: {top_25.shape}")
print()
print(top_25.select(["scopus_id", "title", "predicted_probability"]))

# Save submission file
output_path = "top_25_predictions.parquet"
top_25.write_parquet(output_path)
print(f"\nSaved to: {output_path}")
