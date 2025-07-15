import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    'text': [
        "Free money offer!",
        "Hi friend, how are you?",
        "Congratulations, you won a prize!",
        "Let's meet for lunch",
        "Win a free ticket now!",
        "Can we reschedule the meeting?",
        "Click here to claim your reward",
        "How was your day?"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

X = df['text']
y = df['label']
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
