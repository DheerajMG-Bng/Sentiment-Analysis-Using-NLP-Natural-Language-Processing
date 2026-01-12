# Sentiment Analysis using NLP (Classical Machine Learning)

 Project Overview

This project implements an **end-to-end sentiment analysis system** using **classical Natural Language Processing (NLP) and Machine Learning techniques**. The goal is to classify **opinion-based text reviews** into **Positive** or **Negative** sentiment.

The project was developed with a strong focus on:

* Correct NLP preprocessing
* Proper feature–model pairing
* Error analysis and iterative improvement
* Honest handling of model limitations

Rather than relying on deep learning, this project demonstrates a **solid foundational understanding of NLP pipelines**, which is highly valued for internships and entry-level ML roles.


 Concepts & Techniques Covered

### 1. Natural Language Processing (NLP)

* Text cleaning and normalization
* Tokenization
* Stopword removal (with **negation preservation**)
* Lemmatization
* Noise removal (HTML tags, URLs, special characters)

### 2. Feature Extraction Techniques

* **Bag of Words (BoW)**
* **TF-IDF (Term Frequency–Inverse Document Frequency)**
* **Word2Vec (custom-trained embeddings)**

### 3. Machine Learning Models

* **Multinomial Naive Bayes** (for BoW and TF-IDF)
* **Logistic Regression** (for Word2Vec embeddings)

### 4. Model Evaluation

* Train–test split
* Accuracy
* Precision, Recall, F1-score
* Confusion matrix
* Class imbalance analysis

### 5. Model Persistence

* Saving and loading trained models using `joblib`
* Reusable inference pipeline

---

 Project Pipeline

1. **Raw Text Input**
2. **Text Preprocessing**

   * Lowercasing
   * HTML and URL removal
   * Special character removal
   * Stopword removal (negation words retained)
   * Lemmatization
3. **Feature Extraction**

   * BoW / TF-IDF / Word2Vec
4. **Model Training**

   * MultinomialNB or Logistic Regression
5. **Evaluation**

   * Accuracy, F1-score, confusion matrix
6. **Prediction / Inference**

---

 Experiments & Results

### Word2Vec + Logistic Regression

* **Training set shape:** (9600, 100)
* **Test set shape:** (2400, 100)
* **Accuracy:** ~70%

#### Classification Report

| Class            | Precision | Recall   | F1-Score | Support |
| ---------------- | --------- | -------- | -------- | ------- |
| Negative (0)     | 0.54      | 0.74     | 0.62     | 803     |
| Positive (1)     | 0.84      | 0.68     | 0.75     | 1597    |
| **Weighted Avg** | **0.74**  | **0.70** | **0.71** | 2400    |

This shows:

* Strong performance on the majority class
* Reasonable recall on the minority (negative) class
* Realistic performance for classical NLP methods

---

 Visualizations

The project includes the following evaluation visualizations:

* Confusion Matrix
* Precision vs Recall (per class)
* Accuracy comparison across BoW, TF-IDF, and Word2Vec

These plots help in understanding class imbalance and model behavior beyond raw accuracy.

---

 Correct Usage Examples

### Positive Sentences

* "I love this product"
* "The service is excellent"
* "I am very happy with the experience"

### Negative Sentences

* "I hate this product"
* "The service quality is poor"
* "This is the worst experience"

---
 Important Limitations

This project **intentionally does NOT attempt** to solve the following:

* Medical or physical condition statements (e.g., "I am not able to walk")
* Factual or neutral sentences
* Sarcasm or irony detection
* Aspect-based sentiment analysis
* Emotion classification

**Reason:**
The model is trained on **opinion-based review data**, and classical NLP techniques cannot reliably infer sentiment from non-opinion or context-heavy statements.

---

 Key Improvements Made During Development

* Fixed incorrect stopword removal that was deleting negation words
* Matched classifiers correctly with feature types
* Replaced GaussianNB with MultinomialNB for text features
* Switched to Logistic Regression for Word2Vec embeddings
* Retrained Word2Vec with proper vocabulary coverage
* Ensured preprocessing consistency between training and inference

These changes significantly improved prediction reliability.

 
Tech Stack

* Python
* NLTK
* Scikit-learn
* Gensim
* Pandas
* Matplotlib

---
