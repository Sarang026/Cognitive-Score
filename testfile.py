import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb  

# STEP 1 - Load your CSV dataset
df = pd.read_csv('test.csv')

# STEP 2 - Quick cleaning (adjust columns as per your dataset)
df = df.dropna()  # drop missing rows for now

# Encode categorical columns if needed
if 'Gender' in df.columns:
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# STEP 3 - Define features & labels
X = df.drop("Label",axis=1, errors='ignore')  # Features
y = df['Label']  # 0 = Normal, 1 = Challenged

# STEP 4 - Split into train & test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simulating transfer learning:
# First train on older kids (example: age >= 15)
if 'Age' not in X_train_full.columns:
    raise ValueError("The 'Age' column is required for transfer learning split.")

X_train_base = X_train_full.loc[X_train_full['Age'] >= 15]
y_train_base = y_train_full.loc[X_train_full['Age'] >= 15]

# Fine-tuning data (younger or challenged kids)
X_train_fine = X_train_full.loc[X_train_full['Age'] < 15]
y_train_fine = y_train_full.loc[X_train_full['Age'] < 15]

# STEP 5 - Train base model
base_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
base_model.fit(X_train_base, y_train_base)

# Evaluate base model before transfer
print("Base model evaluation (before transfer):")
y_pred_test = base_model.predict(X_test)
print(classification_report(y_test, y_pred_test))

# STEP 6 - Transfer learning (continue training)
base_model.fit(X_train_fine, y_train_fine, xgb_model=base_model.get_booster())

# Evaluate after transfer
print("Model evaluation after transfer learning:")
y_pred_test_transfer = base_model.predict(X_test)
print(classification_report(y_test, y_pred_test_transfer))

# Random Forest Model for feature importance
# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_full, y_train_full)

# Feature importances
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature importances from the Random Forest model:")
print(importances.sort_values(ascending=False))