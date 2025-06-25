import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------------------------
# 1. Load and Clean the Dataset
# ---------------------------------------------
df = pd.read_csv("human_cognitive_performance.csv")
df = df.drop_duplicates()
df = df.drop(['User_ID', 'AI_Predicted_Score'], axis=1)

# ---------------------------------------------
# 2. Split Features and Target
# ---------------------------------------------
X = df.drop('Cognitive_Score', axis=1)
y = df['Cognitive_Score']

# Identify categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# ---------------------------------------------
# 3. Preprocessing (One-hot encoding)
# ---------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# ---------------------------------------------
# 4. Create Full Pipeline and Train
# ---------------------------------------------
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

pipeline.fit(X, y)

# ---------------------------------------------
# 5. Evaluate the Model
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print("\n📊 Final Linear Regression Model Performance:")
print(f"  🔹 MAE  : {mae:.2f}")
print(f"  🔹 RMSE : {rmse:.2f}")
print(f"  🔹 R²   : {r2:.4f}")

# ---------------------------------------------
# 6. Save the Trained Model
# ---------------------------------------------
joblib.dump(pipeline, "cognitive_score_predictor.pkl")
print("\n✅ Model saved as cognitive_score_predictor.pkl")

# ---------------------------------------------
# 7. Make a Prediction on New User Data
# ---------------------------------------------
new_data = pd.DataFrame([{
    "Age": 26,
    "Gender": "Male",
    "Sleep_Duration": 7.5,
    "Stress_Level": 3,
    "Diet_Type": "Vegetarian",
    "Daily_Screen_Time": 5.0,
    "Exercise_Frequency": "High",
    "Caffeine_Intake": 100,
    "Reaction_Time": 320.0,
    "Memory_Test_Score": 70
}])

# Load the model (optional, just to simulate real-world usage)
model = joblib.load("cognitive_score_predictor.pkl")
prediction = model.predict(new_data)

print(f"\n🧠 Predicted Cognitive Score for New User: {prediction[0]:.2f}")
