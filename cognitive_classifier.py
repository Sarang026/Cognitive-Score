# cognitive_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ========== STEP 1: Load Data ==========
df = pd.read_csv('student_data.csv')  # Make sure this CSV is in the same folder

# ========== STEP 2: Create Cognitive Score ==========
# Weighted formula: 60% test average + 40% GPA (scaled to 100)
df['CognitiveScore'] = (df['TestScore_Math'] + df['TestScore_Reading'] + df['TestScore_Science']) / 3 * 0.6 + df['GPA'] * 25 * 0.4

# Convert CognitiveScore to classification labels: Low / Medium / High
bins = [0, 60, 75, 100]
labels = ['Low', 'Medium', 'High']
df['CognitiveLabel'] = pd.cut(df['CognitiveScore'], bins=bins, labels=labels)

# ========== STEP 3: Encode Categorical Variables ==========
categorical_cols = ['Gender', 'Race', 'ParentalEducation', 'SchoolType', 'Locale']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ========== STEP 4: Prepare Features and Labels ==========
X = df.drop(columns=['CognitiveScore', 'CognitiveLabel'])
y = df['CognitiveLabel']

# ========== STEP 5: Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== STEP 6: Decision Tree Classifier ==========
print("\n--- Decision Tree Classifier ---")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(classification_report(y_test, y_pred_dt))

# ========== STEP 7: Random Forest Classifier ==========
print("\n--- Random Forest Classifier ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# ========== STEP 8: Feature Importance (Random Forest) ==========
importances = rf.feature_importances_
feat_names = X.columns
feat_importance = pd.Series(importances, index=feat_names).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_importance.tail(10).plot(kind='barh', color='teal')
plt.title("Top 10 Important Features - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")  # Saves the plot as an image
plt.show()

# ========== STEP 9: Confusion Matrix (Optional) ==========
cm = confusion_matrix(y_test, y_pred_rf, labels=['Low', 'Medium', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
