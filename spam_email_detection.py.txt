# Spam Email Detection
# Skills: Python, NLP, Scikit-learn, Pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'Email': [
        'Congratulations! You won a free iPhone',
        'Claim your cash prize now',
        'Meeting scheduled for tomorrow',
        'Project report submission deadline',
        'Limited offer buy now',
        'Please send the assignment'
    ],

    'Label': [
        'Spam',
        'Spam',
        'Ham',
        'Ham',
        'Spam',
        'Ham'
    ]
}

# Create dataframe
df = pd.DataFrame(data)

# Features and target
X = df['Email']
y = df['Label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create NLP + ML model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Test prediction
new_email = ["Win ₹50,000 cash reward now"]

prediction = model.predict(new_email)

print("Prediction:", prediction[0])

# Accuracy check
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))