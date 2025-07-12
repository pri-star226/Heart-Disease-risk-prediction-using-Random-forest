❤️ Heart Disease Prediction Dataset
🩺 Overview

This dataset is used to predict the presence of heart disease based on various patient attributes and medical measurements. It is ideal for building classification models in healthcare analytics and clinical decision support systems.

📁 Dataset Summary

Records: 303 patient entries

Features: 13 predictors + 1 target variable

Goal: Predict target (1 = heart disease, 0 = no heart disease)

🔑 Column Descriptions
Feature	Description
age	Age of the patient (years)
sex	Gender (1 = Male, 0 = Female)
cp	Chest pain type (0–3)
trestbps	Resting blood pressure (mm Hg)
chol	Serum cholesterol (mg/dl)
fbs	Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
restecg	Resting electrocardiographic results (0–2)
thalach	Maximum heart rate achieved
exang	Exercise-induced angina (1 = Yes, 0 = No)
oldpeak	ST depression induced by exercise
slope	Slope of the peak exercise ST segment (0–2)
ca	Number of major vessels colored by fluoroscopy (0–3)
thal	Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
target	Target Variable (1 = presence of heart disease, 0 = absence)

🔍 Example Rows
age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
63	1	3	145	233	1	0	150	0	2.3	0	0	1	1
41	0	1	130	204	0	0	172	0	1.4	2	0	2	1

📈 Use Cases
Train ML models (e.g., logistic regression, decision tree, random forest)

Predict patient heart disease risk

Analyze correlations between features (e.g., age vs. heart rate)

Build healthcare dashboards or clinical decision tools

⚙️ Getting Started
Load Dataset in Python
python
Copy
Edit
import pandas as pd

df = pd.read_csv('heart_disease.csv')
print(df.head())
Train a Simple Classifier
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Test Accuracy:", accuracy)
🛠️ Suggested Improvements
Data cleaning and outlier removal

Feature importance visualization

ROC curve and confusion matrix analysis

Web-based prediction using Streamlit or Flask

📜 License
This dataset is made available for educational and research purposes.
