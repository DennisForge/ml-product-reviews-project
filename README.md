# 🧠 Sentiment Analysis ML Project (Complete Pipeline)

A machine learning project that classifies customer product reviews into sentiment categories using natural language processing and multiple classification algorithms.

## Overview

This project analyzes a dataset of 170,000+ product reviews to predict sentiment (positive, negative, neutral) based on review text and metadata. The goal is to compare different ML models and identify the best performing approach for sentiment classification.

## Dataset

- **Source**: Customer product reviews dataset
- **Size**: 170,956 rows (34,191 after cleaning)
- **Features**:
  - `review_title` - Short review summary
  - `review_text` - Full review content
  - `product_price` - Product price in USD
  - `sentiment` - Target variable (positive/negative/neutral)

## Project Structure

```
ml-product-reviews-project/
├── data/                           # Dataset files
├── notebook/
│   └── product_reviews_analysis.ipynb  # Full analysis workflow
├── results/
│   └── model_comparison.txt        # Model evaluation results
└── src/
    ├── train_model.py              # Model training script
    └── test_model.py               # Model testing utilities
```

## Methodology

### Data Preprocessing
1. **Cleaning**: Removed missing values and invalid entries
2. **Text Processing**: Converted review text using TF-IDF vectorization
3. **Feature Engineering**: Created `review_length` feature from review text
4. **Standardization**: Normalized sentiment labels and scaled numeric features

### Models Evaluated

Five classification algorithms were trained and compared:

| Model | Accuracy | Best For |
|-------|----------|----------|
| **Logistic Regression** | 90% | Overall performance |
| **Random Forest** | 90% | Feature importance |
| **Support Vector Machine** | 89% | High-dimensional data |
| **Naive Bayes** | 87% | Fast predictions |
| **Decision Tree** | 87% | Interpretability |

### Feature Pipeline
- **Text features**: TF-IDF vectorization of review title and text
- **Numeric features**: MinMaxScaler for review length
- **Combined**: ColumnTransformer for unified preprocessing

## Results

**Best Models**: Logistic Regression and Random Forest both achieved **90% accuracy** on the test set.

### Key Findings
- Positive reviews are easiest to classify (95% F1-score)
- Neutral reviews pose the biggest challenge (60% F1-score)
- Review text is the strongest predictor of sentiment
- Product price has minimal impact on sentiment classification

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DennisForge/ml-product-reviews-project.git
cd ml-product-reviews-project
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

## Usage

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook notebook/product_reviews_analysis.ipynb
```

### Option 2: Python Script
```bash
python src/train_model.py
```

Results will be saved to `results/model_comparison.txt`.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- jupyter

## Performance

Training all 5 models takes approximately **2 minutes** on MacBook Pro M4 Pro.

## License

This project is created for educational purposes as part of the Introduction to Machine Learning course.

## Author

Denis - [GitHub](https://github.com/DennisForge)