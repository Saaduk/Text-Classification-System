📄 Text Classification System

AI/ML Engineer Intern – Technical Assignment (Ardentix)


📌 Project Overview

In this project, I implemented a Text Classification System that can classify SMS messages as Spam or Ham (Not Spam). My goal was to demonstrate a complete machine learning workflow, starting from raw text data, through preprocessing and feature extraction, to model training and evaluation.

This project shows how natural language processing (NLP) techniques and machine learning algorithms can be applied to solve real-world text classification problems.


🎯 Objective

The main objectives of this project were to:

Understand and preprocess raw text data

Convert text into numerical features suitable for machine learning

Train a classification model to predict categories accurately

Evaluate model performance using standard metrics and analyze results


📊 Dataset

I used the SMS Spam Collection Dataset, which contains 5,574 SMS messages.
The dataset has two labels:

spam – Unwanted or promotional messages

ham – Legitimate messages

This dataset is commonly used for text classification tasks and is ideal for demonstrating NLP workflows.


🧠 My Approach & Methodology
1️⃣ Text Preprocessing

I applied several preprocessing steps to prepare the text for modeling:

Removed special characters and numbers

Converted all text to lowercase

Tokenized the text into words

Removed common stopwords using NLTK

Applied stemming using the Porter Stemmer

These steps helped reduce noise and improved the quality of the input data.

2️⃣ Feature Extraction

I used TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical features.

TF-IDF represents how important a word is in a document relative to the dataset

This method reduces the impact of common but less informative words

It produces a sparse matrix suitable for text classification algorithms

3️⃣ Model Selection

I chose Multinomial Naive Bayes for this task.

Why Naive Bayes?

Works efficiently with high-dimensional, sparse text data

Fast to train and predict

Widely used for text classification problems, including spam detection

4️⃣ Model Training

I split the dataset into 75% training and 25% testing

Trained the model on the training set

Made predictions on the unseen test set

Evaluated the results using accuracy, precision, recall, and F1-score


📈 Model Evaluation

The model performed well on the test data. Here are the metrics I observed:

Metric	Value
Accuracy	96.84%
Precision	99.33%
Recall	77.49%
F1-Score	87.06%

🔍 Observations

The model achieves very high precision, meaning most messages flagged as spam are actually spam.

The recall is lower, which means some spam messages are missed.

This trade-off is acceptable for spam detection because avoiding false positives (marking important messages as spam) is critical.

Overall, the model is reliable and demonstrates a complete ML pipeline.


🛠️ Technologies & Libraries

Python – programming language

NumPy – numerical computing

Pandas – data manipulation

Scikit-learn – machine learning models and evaluation

NLTK – natural language processing

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Program
python Text_classifier.py

📁 Project Structure

text-classification-ardentix/
│
├── data/
│   └── spam.csv
├── Text_classifier.py
├── requirements.txt
└── README.md

🚀 Conclusion

This project demonstrates a complete text classification workflow using machine learning. It shows my understanding of preprocessing, feature extraction, model selection, and evaluation. The project provides insights into handling real-world text data and preparing a model for practical applications like spam detection.

👤 Author

Saad Ullah Khan    
AI/ML Engineer Intern Applicant – Ardentix
