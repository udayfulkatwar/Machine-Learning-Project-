Sure! Here’s your content formatted perfectly for **GitHub README.md** — you can copy and paste it directly:

---

# 🧠 Stock Sentiment Analysis with News Headlines

## 📘 Overview

This project implements machine learning models to predict **stock market sentiment (positive/negative movement)** based on daily news headlines. By analyzing the relationship between news content and market movements, we aim to provide insights for investors and financial analysts on how media narratives influence stock price behavior.

**Key Results:**

* 🌲 **Random Forest Classifier** achieved **84.92% accuracy**
* 🧮 **Multinomial Naive Bayes** achieved **84.66% accuracy**
* 💹 Both models demonstrated strong performance in identifying positive sentiment (**95%+ recall**)

---

## 📊 Dataset

### Source

**Dataset:** [News Headlines Dataset For Stock Sentiment Analyze](https://www.kaggle.com/datasets/siddharthtyagi/news-headlines-dataset-for-stock-sentiment-analyze)
**Author:** Siddharth Tyagi
**Version:** 1
**Platform:** Kaggle Hub

### Data Characteristics

* **Size:** 4,101 entries (days of stock market data)
* **Features:** 27 columns total

  * 1 Date column
  * 1 Label column (0 = negative/neutral, 1 = positive sentiment)
  * 25 headline columns (Top1 through Top25 – daily news headlines)
* **Training Set:** 3,975 rows (data before January 1, 2015)
* **Testing Set:** 378 rows (data after December 31, 2014)

### Preprocessing Steps

1. **Text Cleaning:** Removed non-alphabetic characters (numbers, punctuation) using regex patterns and replaced them with spaces to standardize the text.
2. **Headline Aggregation:** Combined all 25 daily headlines into a single text string per day, creating a comprehensive daily news narrative.
3. **Feature Extraction:** Applied `CountVectorizer` with **bigram features (ngram_range=(2,2))** to capture meaningful word pairs rather than individual words.
4. **Temporal Splitting:** Used chronological split (before/after 2015) to avoid data leakage and simulate real-world prediction scenarios.

---

## ⚙️ Methods

### Approach

We implemented two complementary machine learning algorithms for binary sentiment classification:

#### 1. **Random Forest Classifier**

* **Configuration:** 200 estimators, entropy criterion, `random_state=42`
* **Rationale:**

  * Handles high-dimensional text features effectively
  * Resistant to overfitting through ensemble learning
  * Captures complex, non-linear relationships between word patterns and sentiment
  * No strong independence assumptions required

#### 2. **Multinomial Naive Bayes Classifier**

* **Configuration:** Default parameters (`alpha=1.0`)
* **Rationale:**

  * Established baseline for text classification tasks
  * Computationally efficient
  * Works well with word frequency features despite independence assumption
  * Provides probabilistic interpretation of predictions

### Why This Approach?

**Feature Representation (Bigrams):**

* Bigrams capture contextual word pairs (e.g., “stock rose”, “market fell”) that are more informative than single words for sentiment analysis.
* Preserves some semantic meaning that unigrams might miss.

**Model Selection:**

* **Random Forest:** Captures complex feature interactions in news language.
* **Naive Bayes:** Efficient baseline validating the Random Forest performance.
* Both models are interpretable and suitable for structured text features.

### Alternative Approaches Considered

| Method                            | Why Not Chosen                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| **Support Vector Machines (SVM)** | Computationally expensive for high-dimensional text data (~100K+ bigram features)         |
| **Deep Learning (LSTM/BERT)**     | Requires more computational resources, larger datasets, and complex hyperparameter tuning |
| **Unigram Features**              | Less informative than bigrams for capturing sentiment context                             |

### Pipeline Diagram

```
[Raw Data: News Headlines + Labels]
           ↓
[Text Preprocessing]
  - Remove special characters
  - Combine 25 headlines per day
           ↓
[Feature Extraction]
  - CountVectorizer (bigrams)
  - Sparse matrix representation
           ↓
[Model Training]
  - Random Forest (200 trees)
  - Multinomial Naive Bayes
           ↓
[Prediction & Evaluation]
  - Confusion Matrix
  - Accuracy, Precision, Recall, F1-Score
```

---

## 🧩 Installation & Setup

### Prerequisites

* Python 3.7+
* pip package manager

### Required Libraries

```bash
pip install pandas scikit-learn kagglehub
```

### Steps to Run

1. **Clone the repository:**

```bash
git clone <repository-url>
cd stock-sentiment-analysis
```

2. **Run the analysis:**

```bash
python sentiment_analysis.py
```

Or execute the Jupyter notebook:

```bash
jupyter notebook ML.ipynb
```

3. **Output:**
   The script will:

* Automatically download the dataset from Kaggle Hub
* Preprocess the text data
* Train both models
* Display performance metrics (confusion matrix, accuracy, classification report)

---

## 📈 Experiments & Results

### Performance Comparison

| Metric                  | Random Forest | Multinomial Naive Bayes |
| ----------------------- | ------------- | ----------------------- |
| **Accuracy**            | **84.92%**    | 84.66%                  |
| **Precision (Class 0)** | 0.94          | 0.93                    |
| **Recall (Class 0)**    | 0.74          | 0.74                    |
| **F1-Score (Class 0)**  | 0.83          | 0.83                    |
| **Precision (Class 1)** | 0.79          | 0.79                    |
| **Recall (Class 1)**    | **0.96**      | **0.95**                |
| **F1-Score (Class 1)**  | **0.87**      | 0.86                    |

### 🟩 Random Forest Classifier Results

**Confusion Matrix:**

```
                 Predicted Negative  Predicted Positive
Actual Negative         137                 49
Actual Positive           8                184
```

**Key Observations:**

* High recall for positive sentiment (0.96)
* Lower recall for negative sentiment (0.74)
* Conservative with negative predictions, confident with positives

### 🟦 Multinomial Naive Bayes Results

**Confusion Matrix:**

```
                 Predicted Negative  Predicted Positive
Actual Negative         138                 48
Actual Positive          10                182
```

**Key Observations:**

* Nearly identical accuracy to Random Forest (84.66%)
* More balanced predictions between classes
* Excellent generalization for a simple model

### Model Insights

#### Strengths

1. High positive sentiment recall (95%+)
2. Minimal performance difference between models
3. Consistent test set generalization

#### Limitations

1. Some false positives (~26% for class 0)
2. Slight bias toward positive sentiment
3. Temporal dependencies not modeled

---

## 🎯 Conclusion

### Key Findings

1. **News Headlines are Predictive:** Achieved ~85% accuracy using only textual data.
2. **Bigrams are Powerful:** Capture sentiment-rich word combinations.
3. **Model Parity:** Naive Bayes nearly matches Random Forest — simple models suffice with good features.
4. **High Positive Recall:** Valuable for bullish trading signals.

### Practical Implications

* Useful for identifying **positive market sentiment days**.
* Exercise caution with bearish predictions (higher false positives).
* Naive Bayes is more computationally efficient for deployment.

### Limitations & Future Work

* Does not capture **time-series patterns**.
* Binary classification ignores **movement magnitude**.
* Only considers **news headlines**, not market indicators.

**Future Enhancements:**

1. Introduce LSTM/BERT for temporal modeling
2. Expand to multi-class classification
3. Combine with technical indicators
4. Enable real-time prediction via API
5. Explore fine-tuned FinBERT for finance-specific sentiment

### Lessons Learned

* Classic ML + good features = strong results
* Feature engineering is critical
* Simplicity can rival complexity
* Chronological splits prevent data leakage

---

## 📚 References

1. **Dataset:**
   Tyagi, S. (2024). *News Headlines Dataset For Stock Sentiment Analyze* (Version 1). Kaggle.
   [Link](https://www.kaggle.com/datasets/siddharthtyagi/news-headlines-dataset-for-stock-sentiment-analyze)

2. **Libraries & Tools:**

   * Scikit-learn: Pedregosa et al. (2011). *JMLR*, 12, 2825–2830.
   * KaggleHub API: [kaggle.com/docs/api](https://www.kaggle.com/docs/api)
   * Pandas: McKinney, W. (2010). *Python in Science Conference*.

3. **Methodological References:**

   * Breiman, L. (2001). *Random Forests*, Machine Learning, 45(1), 5–32.
   * Rennie, J. D., et al. (2003). *ICML*, 3, 616–623.

---

## 📂 Project Structure

```
stock-sentiment-analysis/
├── ML.ipynb                 # Jupyter notebook with full analysis
├── ML.html                  # HTML export of notebook
├── sentiment_analysis.py    # Python script version
├── ML_Project.pdf           # Project report
├── README.md                # This file
└── data/                    # Auto-downloaded via kagglehub
    └── Data.csv
