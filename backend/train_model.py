
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
from gesture_features import FEATURES_FOR_TWO_HANDS, FEATURES_PER_HAND

DATA_PATH = "backend/data/gesture_samples.csv"
MODEL_PATH = "backend/models/gesture_model.pkl"

# 1. Load dataset
X = []
y = []
X_two_hands = []
y_two_hands = []
skipped_rows = 0
expected = ["label"] + [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

with open(DATA_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader, None)
    if header is None or header[:len(expected)] != expected:
        raise ValueError("Dataset columns are incorrect. Check CSV header.")

    for row in reader:
        if not row:
            continue
        label = row[0].strip()
        if not label:
            continue

        raw_features = row[1:]
        if len(raw_features) not in {FEATURES_PER_HAND, FEATURES_FOR_TWO_HANDS}:
            skipped_rows += 1
            continue

        try:
            features = [float(value) for value in raw_features[:FEATURES_FOR_TWO_HANDS]]
        except (TypeError, ValueError):
            skipped_rows += 1
            continue

        if not np.isfinite(features).all():
            skipped_rows += 1
            continue

        X.append(features[:FEATURES_PER_HAND])
        y.append(label)
        if len(features) == FEATURES_FOR_TWO_HANDS:
            X_two_hands.append(features)
            y_two_hands.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)
X_two_hands = np.asarray(X_two_hands, dtype=np.float32)
y_two_hands = np.asarray(y_two_hands)

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
print("Single-hand feature count:", X.shape[1])
print("True two-hand samples:", len(X_two_hands))
print("Skipped malformed rows:", skipped_rows)

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

two_hand_counts = Counter(y_two_hands)
minimum_two_hand_samples = 5
eligible_two_hand_labels = {
    label
    for label, count in two_hand_counts.items()
    if count >= minimum_two_hand_samples
}
two_hand_mask = np.isin(y_two_hands, list(eligible_two_hand_labels))
X_two_train = X_two_hands[two_hand_mask]
y_two_train = y_two_hands[two_hand_mask]

two_hand_model = None
if len(eligible_two_hand_labels) >= 2:
    print("\nTraining two-hand Random Forest model...")
    two_hand_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    two_hand_model.fit(X_two_train, y_two_train)
else:
    print("Not enough two-hand classes to train a separate two-hand model.")

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
    "feature_count": X.shape[1],
    "hands_supported": 1,
    "two_hand_model": two_hand_model,
    "two_hand_classes": (
        two_hand_model.classes_.tolist() if two_hand_model is not None else []
    ),
    "two_hand_feature_count": (
        FEATURES_FOR_TWO_HANDS if two_hand_model is not None else 0
    ),
    "two_hand_class_counts": dict(two_hand_counts),
    "minimum_two_hand_samples": minimum_two_hand_samples,
}, MODEL_PATH)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Location:", MODEL_PATH)
