import re
import numpy as np
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

# ===============================
# Download NLTK data (first time only)
# ===============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# Load saved models
# ===============================
lr_w2v = joblib.load("w2v_lr_model.pkl")   # Logistic Regression model
w2v_model = Word2Vec.load("w2v_model.model")

# ===============================
# Preprocessing (NEGATION SAFE)
# ===============================
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard('nor')
stop_words.discard('never')

def preprocess(text):
    text = str(text).lower()
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub('[^a-z ]', '', text)

    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ]

    return " ".join(words)

# ===============================
# Sentence → Word2Vec vector
# ===============================
def document_vector(tokens, model):
    tokens = [w for w in tokens if w in model.wv.key_to_index]
    if len(tokens) == 0:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[tokens], axis=0)

# ===============================
# Prediction function
# ===============================
def predict_sentence(sentence):
    clean_text = preprocess(sentence)
    tokens = word_tokenize(clean_text)

    vector = document_vector(tokens, w2v_model).reshape(1, -1)
    prediction = lr_w2v.predict(vector)[0]

    return "Positive" if prediction == 1 else "Negative"

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    text = input("Enter a sentence for Sentiment Analysis: ")
    print("Predicted Sentiment:", predict_sentence(text))
