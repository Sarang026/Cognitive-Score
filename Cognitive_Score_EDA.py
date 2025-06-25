import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('human_cognitive_performance.csv')

# --- BASIC INFO ---
print("Shape:", df.shape)
print("\nColumn Info:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# --- UNIVARIATE ANALYSIS ---
# Histograms for numerical features
df.hist(figsize=(15, 12), bins=30)
plt.suptitle("Distributions of Numerical Features")
plt.tight_layout()
plt.show()

# Count plots for categorical features
categorical_columns = ['Gender', 'Diet_Type', 'Exercise_Frequency']
for col in categorical_columns:
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()

# --- BIVARIATE ANALYSIS ---
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots for Cognitive Score by Category
for col in categorical_columns:
    sns.boxplot(data=df, x=col, y='Cognitive_Score')
    plt.title(f'Cognitive Score by {col}')
    plt.xticks(rotation=45)
    plt.show()

# Scatter Plots
sns.scatterplot(x='Age', y='Cognitive_Score', data=df)
plt.title("Age vs Cognitive Score")
plt.show()

sns.scatterplot(x='Sleep_Duration', y='Cognitive_Score', data=df)
plt.title("Sleep Duration vs Cognitive Score")
plt.show()

sns.scatterplot(x='Daily_Screen_Time', y='Cognitive_Score', data=df)
plt.title("Screen Time vs Cognitive Score")
plt.show()
