import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1️⃣ Download stopwords
nltk.download('stopwords')

# 2️⃣ Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 3️⃣ Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 4️⃣ Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
corpus = []

for text in df['message']:
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove numbers and special chars
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    corpus.append(" ".join(words))

# 5️⃣ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(corpus).toarray()
y = df['label']

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 7️⃣ Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 8️⃣ Predictions
y_pred = model.predict(X_test)

# 9️⃣ Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# 🔹 10️⃣ Save trained model and vectorizer
import os
if not os.path.exists('model'):
    os.makedirs('model')

pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))
print("Model and vectorizer saved successfully in 'model/' folder.")
