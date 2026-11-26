
import csv
import os
import joblib
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "backend/data/gesture_samples.csv"
MODEL_PATH = "backend/models/gesture_model.pkl"

# 1. Load dataset
X = []
y = []

with open(DATA_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    expected = ["label"] + [
        f"{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

    if reader.fieldnames != expected:
        raise ValueError("Dataset columns are incorrect. Check CSV header.")

    for row in reader:
        label = row["label"].strip()

        if not label:
            continue

        features = [
            float(row[column])
            for column in expected[1:]
        ]

        if not np.isfinite(features).all():
            continue

        X.append(features)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# 2. Check dataset
counts = Counter(y)

print("\nGESTURE SAMPLE COUNTS:")
for label, count in sorted(counts.items()):
    print(f"{label}: {count}")

if len(counts) < 2:
    raise ValueError("At least two gesture classes are required.")

if min(counts.values()) < 2:
    raise ValueError("Each gesture needs at least two valid samples.")

print("\nTotal samples:", len(X))
print("Number of gestures:", len(counts))
print("Feature count:", X.shape[1])

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Train Random Forest
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
))

# 6. Save trained model
os.makedirs("backend/models", exist_ok=True)

joblib.dump({
    "model": model,
    "classes": model.classes_.tolist(),
    "feature_count": X.shape[1]
}, MODEL_PATH)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Location:", MODEL_PATH)
