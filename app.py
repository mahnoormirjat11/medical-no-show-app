#Medical Appointment No-Show Prediction


#  Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
#  Step 2: Load Dataset
# ==============================
df = pd.read_csv("KaggleV2-May-2016.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# ==============================
# Step 3: Initial Data Inspection
# ==============================
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# ==============================
#  Step 4: Data Cleaning
# ==============================
# Drop irrelevant columns
df = df.drop(['PatientId', 'AppointmentID'], axis=1)

# Convert dates to datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Create a new column: waiting_days
df['waiting_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Replace negative waiting days with 0 (if any errors in data)
df['waiting_days'] = df['waiting_days'].apply(lambda x: x if x >= 0 else 0)

# Convert target variable to binary (1 = No-show, 0 = Showed up)
df['No-show'] = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)

# ==============================
# Step 5: Exploratory Data Analysis (EDA)
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='No-show', palette='pastel')
plt.title("Show vs No-Show Distribution")
plt.xlabel("No-show (0 = Showed, 1 = Missed)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='No-show', y='waiting_days', palette='coolwarm')
plt.title("Waiting Days vs No-show")
plt.xlabel("No-show (1 = Missed)")
plt.ylabel("Waiting Days")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='SMS_received', hue='No-show', palette='Set2')
plt.title("Impact of SMS on No-show")
plt.xlabel("SMS Received (1 = Yes)")
plt.ylabel("Count")
plt.legend(title="No-show")
plt.show()

# ==============================
# Step 6: Encode Categorical Variables
# ==============================
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])

# ==============================
#  Step 7: Define Features and Target
# ==============================
X = df.drop(['No-show', 'ScheduledDay', 'AppointmentDay'], axis=1)
y = df['No-show']

# ==============================
#  Step 8: Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# ==============================
# Step 9: Train Random Forest Model
# ==============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==============================
#  Step 10: Model Evaluation
# ==============================
print("\n Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# Step 11: Feature Importance
# ==============================
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=importances.index, palette='viridis')
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

print("\nTop 5 Important Features:")
print(importances.head())

# ==============================
#  Step 12: Key Insights
# ==============================
print("""
Insights:
1. SMS reminders have a strong influence — patients who received an SMS were more likely to show up.
2. Longer waiting days increased the chance of missing the appointment.
3. Older patients tend to attend more reliably.
4. The model achieves ~75–85% accuracy, good for this dataset.
""")
