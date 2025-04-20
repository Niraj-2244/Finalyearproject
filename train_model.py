import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths to CSV files
casing_csv = "casing_quality_data.csv"
disease_csv = "disease_data.csv"
harvest_csv = "harvest_data.csv"

# Output directory
os.makedirs("./models", exist_ok=True)

# --- 1. Train Casing Quality Model ---
casing_data = pd.read_csv("C:/Users/sahni/Downloads/casing_quality_data.csv")
X_casing = casing_data[['pH', 'WHC', 'Temperature', 'Humidity', 'EC']]
y_casing = casing_data['Casing_Quality']

model_casing = RandomForestClassifier()
model_casing.fit(X_casing, y_casing)

with open("./models/casing_quality.pkl", "wb") as f:
    pickle.dump(model_casing, f)

# Data Visualization
sns.pairplot(casing_data, hue="Casing_Quality")
plt.suptitle("Casing Quality Feature Distribution", y=1.02)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(casing_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# --- 2. Train Disease Detection Model ---
disease_data = pd.read_csv("C:/Users/sahni/Downloads/disease_data.csv")
X_disease = disease_data[['pH', 'Temperature', 'Humidity', 'Ventilation', 'Light_Intensity', 'CO2_Level']]
y_disease = disease_data['Disease_Risk']

preproc_disease = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Ventilation'])
], remainder='passthrough')

with open("./models/disease_detection.pkl", "wb") as f:
    pickle.dump({
        'model': model_disease,
        'preprocessor': preproc_disease
    }, f)

# Data Visualization
sns.pairplot(disease_data, hue="Disease")
plt.suptitle("Disease Detection Feature Distribution", y=1.02)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(disease_data.corr(), annot=True, cmap="YlGnBu")
plt.title("Disease Feature Correlation Heatmap")
plt.show()

# --- 3. Train Harvest Prediction Model ---
harvest_data = pd.read_csv("C:/Users/sahni/Downloads/harvest_data.csv")
X_harvest = harvest_data[['Bags', 'Spawn_Type', 'Casing_Type', 'Days_Since_Casing']]
y_harvest = harvest_data['Harvest_kg']

preproc_harvest = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Spawn_Type', 'Casing_Type'])
], remainder='passthrough')

X_harvest_transformed = preproc_harvest.fit_transform(X_harvest)
model_harvest = RandomForestRegressor()
model_harvest.fit(X_harvest_transformed, y_harvest)

with open("./models/harvest_prediction.pkl", "wb") as f:
    pickle.dump({
        'model': model_harvest,
        'preprocessor': preproc_harvest
    }, f)

# Visualization: Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True, bins=30, color='skyblue')
plt.title("Distribution of Harvest (Kg)")
plt.xlabel("Harvest (Kg)")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(harvest_data.corr(), annot=True, cmap="coolwarm")
plt.title("Harvest Feature Correlation Heatmap")
plt.show()

print("✅ Models trained and saved to ./models")
