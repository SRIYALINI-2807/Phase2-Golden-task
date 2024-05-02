
# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Data (replace with your dataset)
data = {
    'text': [
        'buy now, discount offer',  # spam
        'hello, how are you',  # ham
        'free gift, claim now',  # spam
        'meeting scheduled for tomorrow',  # ham
        'urgent: call now'  # spam
    ],
    'label': [1, 0, 1, 0, 1]  # 1 for spam, 0 for ham
}
df = pd.DataFrame(data)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Feature Extraction
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Training the Model
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Evaluating the Model
X_test_counts = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization
labels = ['ham', 'spam']
fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap='Blues')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
