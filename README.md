# Emotion Detection in Conversation using NLP 😊🤔😒
## 📌 Project Overview
This project focuses on developing a system that can automatically detect emotions expressed in conversations using Natural Language Processing (NLP) and the Naive Bayes classifier. The goal is to enhance human-computer interaction by enabling machines to interpret the emotional tone of user conversations and respond accordingly.

By analyzing dialogues, the system identifies sentiments such as positive, negative, or neutral, and classifies them into specific emotions like joy, sadness, anger, surprise, and more.

## 🎯 Objectives
* 🎭 Accurately classify emotional tones from text data.

* ⚡ Enable real-time emotion detection for conversational AI (chatbots, assistants).

* 🌐 Support cross-domain use cases like healthcare, social media monitoring, and customer service.

* 🤖 Integrate emotion detection into AI systems for personalized, empathetic responses.

## 🗂️ Dataset Overview
The dataset used is based on annotated dialogues, including fields such as:

* Utterance: Line of dialogue from a speaker

* Speaker: Who said the utterance

* Emotion: Emotion label (e.g., joy, anger, surprise)

* Sentiment: Positive, negative, or neutral

* Dialogue_ID, Utterance_ID, Season, Episode: Metadata for tracking

## 🧠 Algorithm Used
### Naive Bayes Classifier
A probabilistic, lightweight machine learning algorithm that:

* Assumes independence among words

* Calculates probability of emotion based on word frequency (Bag of Words or TF-IDF)

* Ideal for text classification tasks

## ⚙️ Methodology
**1.Data Collection**
  * Dialogues with labeled emotions from conversational data.

**2.Data Preprocessing**

  * Cleaning text (removing special chars, stopwords)

  * Label encoding of emotions/sentiments

  * Vectorization using TF-IDF

**3.Feature Extraction**

  * TF-IDF transforms text into numeric vectors

  * Input features passed into Naive Bayes model

**4.Model Training**

  * 80/20 train-test split

  * Model trained on labeled text data using MultinomialNB

**5.Evaluation Metrics**

  * Accuracy: 90%

  * Confusion Matrix

  * Precision, Recall, F1-score

## 🧪 Implementation
* Language: Python

* Libraries: scikit-learn, pandas, matplotlib, seaborn

* TF-IDF + Naive Bayes Pipeline

* Confusion matrix and sentiment distribution plotted using Matplotlib

## 📊 Model Performance
* Metric	Score
* Accuracy	90%
* Precision	1.00 (negative), 0.80 (neutral), 1.00 (positive)
* Recall	1.00, 1.00, 0.50
* F1 Score	0.89 (avg weighted)

✅ The model effectively detects negative and neutral emotions; performance can be further improved for positive sentiment detection.

## 🔮 Future Scope
* 💻 Upgrade to deep learning models (e.g., LSTM, BERT) for better context awareness.

* 🔁 Handle multi-class and multi-label emotions simultaneously.

* 📲 Deploy in real-time applications like voice assistants and chatbots.

* 🧘 Apply in mental health diagnostics and personalized therapy systems.
