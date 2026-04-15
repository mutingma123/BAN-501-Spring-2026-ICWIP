# BAN-501 Final Project: AI for Consumer Behavior Classification

## Overview

Your task is to build a transformer-based text classifier that identifies academic publications applying artificial intelligence techniques to consumer behavior research. You are given a dataset of over 300,000 publication abstracts from Scopus, collected via an "artificial intelligence" keyword search. The challenge: no labels exist. You must design your own labeling strategy, train a classifier, evaluate its performance, and communicate your methodology and findings.

This project mirrors a common real-world scenario in applied machine learning: you have a large corpus of text and a classification goal, but no ground truth labels. Success requires thoughtful problem definition, creative labeling approaches, and rigorous evaluation.

## Learning Objectives

By completing this project, you will demonstrate:

1. **Problem framing**: Operationalizing an abstract concept ("AI for consumer behavior") into a concrete classification task
2. **Labeling strategy design**: Developing an approach to create training labels when none exist
3. **Transformer fine-tuning**: Applying pre-trained language models to a domain-specific classification task
4. **Evaluation methodology**: Designing appropriate train/validation/test splits and selecting relevant metrics
5. **Error analysis**: Identifying systematic failure modes and understanding model limitations
6. **Technical communication**: Presenting complex methodology clearly to a mixed audience

## Dataset Description

**File**: `data/ai_publications_2020_plus.parquet`

**Records**: 306,866 publication abstracts from 2020-2026

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `scopus_id` | string | Unique Scopus identifier (e.g., "2-s2.0-85186955255") |
| `title` | string | Publication title |
| `abstract` | string | Full abstract text |
| `year` | int | Publication year (2020-2026) |
| `publication_name` | string | Journal or conference name |
| `publication_type` | string | "Journal", "Conference Proceeding", "Book", etc. |
| `subtype` | string | "Article", "Conference Paper", "Review", etc. |
| `doi` | string | DOI if available (may be empty) |
| `keywords` | string | Author-provided keywords, semicolon-separated |

**Loading the data**:
```python
import polars as pl

df = pl.read_parquet("data/ai_publications_2020_plus.parquet")
print(f"Records: {len(df):,}")
print(df.head())
```

## Task Definition

Your classifier should identify publications that apply AI/ML techniques to understand **consumer behavior**. For this project, "consumer behavior" is defined narrowly as research focused on:

- **Purchase decisions**: What, when, why, and how consumers buy
- **Brand choice and preference**: How consumers select among alternatives
- **Consumer psychology**: Attitudes, perceptions, motivations, and decision-making processes
- **Shopping behavior**: In-store and online browsing, cart abandonment, price sensitivity

**Positive examples** (should be classified as consumer behavior):
- "We use deep learning to predict customer purchase intent from browsing sessions..."
- "This study applies sentiment analysis to understand brand perception from social media..."
- "A neural network model for personalized product recommendations based on past purchases..."

**Negative examples** (should NOT be classified as consumer behavior):
- "We propose a CNN for medical image classification..." (AI, but not consumer behavior)
- "This survey examines consumer attitudes toward AI adoption..." (consumer behavior, but not AI applied to it)
- "Deep reinforcement learning for robotic manipulation..." (AI, but not consumer-related)
- "Analysis of consumer spending patterns using regression..." (consumer behavior, but traditional stats not AI)

The boundary cases are where the interesting challenges lie. Your team must make defensible decisions about what counts and document your reasoning.

## Requirements

### 1. Labeling Strategy (25 points)

You must create labeled training data. Possible approaches include (but are not limited to):

- **Keyword-based bootstrapping**: Use keyword patterns to create initial pseudo-labels, then refine
- **Manual labeling**: Hand-label a sample of abstracts (time-intensive but high quality)
- **LLM-assisted labeling**: Use a large language model API to generate candidate labels
- **Active learning**: Iteratively label the most uncertain examples
- **Hybrid approaches**: Combine multiple strategies

Whatever approach you choose, you must:
- Justify why this approach is appropriate for the task
- Acknowledge its limitations and potential biases
- Describe how you handled ambiguous cases

### 2. Model Training (25 points)

**Compute constraint**: This project assumes Google Colab free tier. Models like DistilBERT (~66M parameters) are recommended. Larger models (BERT-base, RoBERTa) may exceed memory or time limits.

You must decide:
- **Which model** to use (DistilBERT, TinyBERT, etc.)
- **Training approach**: Fine-tune only the classification head, or the full model?
- **Hyperparameters**: Learning rate, batch size, epochs, etc.

Either head-only or full fine-tuning is acceptable if you explain the tradeoff. Factors to consider: training time, dataset size, risk of overfitting, expected performance gain.

### 3. Evaluation & Analysis (20 points)

Your evaluation must include:
- **Proper data splits**: Train/validation/test sets with no leakage
- **Relevant metrics**: Precision, recall, F1-score (and why these matter for your use case)
- **Error analysis**: Examine false positives and false negatives. What patterns do you see? Are certain types of papers systematically misclassified?

### 4. Video Presentation (20 points)

Create a **15-minute YouTube video** that includes:
1. **Problem statement**: What are you trying to classify and why does it matter?
2. **Methodology**: How did you create labels? How did you train the model?
3. **Results**: What performance did you achieve? Show metrics and visualizations.
4. **Examples**: Walk through specific true positives, false positives, and false negatives
5. **Reflection**: What worked? What didn't? What would you do differently?

All team members should participate in the presentation.

### 5. Reflection & Future Work (10 points)

In your video, discuss:
- Honest assessment of limitations in your approach
- Concrete ideas for improvement given more time and compute resources
- How you would validate the classifier's real-world utility

## Technical Constraints

- **Compute**: Google Colab free tier (limited GPU time, ~12GB RAM)
- **Recommended model**: `distilbert-base-uncased` or similar small transformer
- **Libraries**: PyTorch, Hugging Face Transformers, scikit-learn, polars/pandas

## Deliverables

### 1. YouTube Video
- Duration: **15 minutes** (videos significantly over/under will lose points)
- Visibility: Unlisted or Public
- Submit: YouTube link

### 2. Top 25 Predictions Parquet
Submit a parquet file containing the 25 abstracts your model is most confident are consumer behavior research.

**Required columns**:
| Column | Type | Description |
|--------|------|-------------|
| `scopus_id` | string | From original dataset |
| `title` | string | From original dataset |
| `abstract` | string | From original dataset |
| `predicted_probability` | float | Model's confidence (0-1) |

The file should be sorted by `predicted_probability` descending (highest confidence first).

See `submission_template.py` for the expected format.

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Labeling Strategy** | 25 | Clear justification for chosen approach. Acknowledgment of limitations. Thoughtful handling of edge cases. Documentation of labeling decisions. |
| **Model Training** | 25 | Appropriate model choice for constraints. Justified training approach (head-only vs full). Reasonable hyperparameter choices. Evidence of iterative refinement. |
| **Evaluation & Analysis** | 20 | Proper train/val/test splits. Appropriate metrics with interpretation. Substantive error analysis with specific examples. |
| **Video Presentation** | 20 | Clear explanation of methodology. Well-organized narrative. Effective visualizations. All team members participate. Stays within 15 minutes. |
| **Reflection & Future Work** | 10 | Honest assessment of what worked and what didn't. Concrete, actionable improvement ideas. Thoughtful consideration of real-world deployment. |
| **Total** | 100 | |

## Timeline

- **Due**: Thursday, April 30, 2026 at 5:00 PM
- Submit: YouTube link + top-25 parquet file

## Team Requirements

- Teams of **up to 3 students**
- All team members must contribute to the video presentation
- Include team member names in the video and submission

## Getting Started

1. Load and explore the dataset (see `load_data_example.py`)
2. Read through several abstracts to understand the classification challenge
3. Develop your labeling strategy and create initial training labels
4. Implement your classifier using the course notebooks as reference (especially `transformer-classification.py`)
5. Train, evaluate, iterate
6. Record and edit your video

## Tips

- **Start with labeling early**: Creating good training data takes longer than you think
- **Keep a decision log**: Document your choices so you can explain them in the video
- **Test on held-out data**: Don't evaluate on data used for any labeling decisions
- **Watch your time**: 15 minutes goes quickly; practice your presentation
- **Show your work**: The process matters as much as the final performance number
