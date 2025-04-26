import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('data.csv', header=None)
data.columns = [f"feature_{i}" for i in range(60)] + ['label']
data['label'] = data['label'].map({'R': 0, 'M': 1})

X = data.drop('label', axis=1)
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.11, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit on the original training data
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM (RBF)': SVC(kernel='rbf', probability=True),
    'KNN': KNeighborsClassifier(),
    'LightGBM': LGBMClassifier(verbose=-1),
    'CatBoost': CatBoostClassifier(verbose=0),
    'HistGradientBoosting': HistGradientBoostingClassifier()
}

# Store results
results = {}

for name, model in models.items():
    # Train on the original scaled training data and labels
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)
    acc_test = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_train_pred)
    prec = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'Test Accuracy': acc_test,
        'Train Accuracy': acc_train,
        'Precision': prec,
        'Confusion Matrix': cm,
        'Model': model
    }

# Display results
res_df = pd.DataFrame({
    model: [res['Train Accuracy'], res['Test Accuracy'], res['Precision']]
    for model, res in results.items()
}, index=['Train Accuracy', 'Test Accuracy', 'Precision']).T

print("\nModel Performance Summary\n")
print(res_df)
sample = np.array([0.037,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055])
sample_scaled = scaler.transform(sample.reshape(1, -1))

# Predict with best model (HistGradientBoosting in this case)
print("\nFinal Prediction for the input sample using all models:\n")
for name, res in results.items():
    pred = res['Model'].predict(sample_scaled)
    label = "Mine" if pred[0] == 1 else "Rock"
    print(f"{name}: {label}")


    # Final prediction using precision-weighted voting
rock_score = 0
mine_score = 0

print("\nFinal Prediction for the input sample using precision-weighted voting:\n")
for name, res in results.items():
    model = res['Model']
    precision = res['Precision']
    prediction = model.predict(sample_scaled)[0]
    label = 'Mine' if prediction == 1 else 'Rock'
    print(f"{name}: {label}")

    if prediction == 1:
        mine_score += precision
    else:
        rock_score += precision

print("\nWeighted Score Summary ")
print(f"Rock Score: {rock_score:.4f}")
print(f"Mine Score: {mine_score:.4f}")

final_prediction = "Mine" if mine_score > rock_score else "Rock"
print(f"\n🎯 Final Weighted Prediction based on Precision: {final_prediction}")


# Confusion Matrix Heatmaps
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, (name, res) in enumerate(results.items()):
    sns.heatmap(res['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=axs[i])
    axs[i].set_title(f"{name}")
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Bar Plot
res_df[['Test Accuracy', 'Precision']].plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Model Comparison - Accuracy & Precision')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()