import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_students = 200
data = {
    'Midterm_Score': np.random.randint(0, 101, num_students),
    'Homework_Average': np.random.randint(0, 101, num_students),
    'Attendance': np.random.randint(0, 101, num_students),
    'At_Risk': np.random.choice([0, 1], num_students, p=[0.8, 0.2]) # 20% at risk
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers etc.)
# For this synthetic data, no cleaning is explicitly needed.
# --- 3. Predictive Modeling ---
X = df[['Midterm_Score', 'Homework_Average', 'Attendance']]
y = df['At_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
# --- 4. Visualization ---
# Feature Importance (Illustrative -  LogisticRegression doesn't directly provide feature importance like tree-based models)
plt.figure(figsize=(8, 6))
plt.bar(['Midterm', 'Homework', 'Attendance'], model.coef_[0])
plt.title('Feature Coefficients (Illustrative)')
plt.ylabel('Coefficient Magnitude')
plt.savefig('feature_coefficients.png')
print("Plot saved to feature_coefficients.png")
#Confusion Matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png")