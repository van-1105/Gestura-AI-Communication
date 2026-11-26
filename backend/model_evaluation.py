
import csv
import json
import os
import time
import joblib
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from gesture_features import FEATURES_FOR_TWO_HANDS, FEATURES_PER_HAND

DATA_PATH = "backend/data/gesture_samples.csv"
MODEL_PATH = "backend/models/gesture_model.pkl"
REPORT_PATH = "backend/models/evaluation_report.json"
TWO_HAND_REPORT_PATH = "backend/models/two_hand_evaluation_report.json"

# 1. Load dataset
X = []
y = []
X_two_hands = []
y_two_hands = []

with open(DATA_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader, None)
    expected = ["label"] + [
        f"{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

    if header is None or header[:len(expected)] != expected:
        raise ValueError("Incorrect dataset columns.")

    for row in reader:
        if not row:
            continue
        label = row[0].strip()
        if not label:
            continue

        raw_features = row[1:]
        if len(raw_features) not in {FEATURES_PER_HAND, FEATURES_FOR_TWO_HANDS}:
            continue

        try:
            features = [float(value) for value in raw_features[:FEATURES_PER_HAND]]
        except (TypeError, ValueError):
            continue

        if np.isfinite(features).all():
            X.append(features)
            y.append(label)
            if len(raw_features) >= FEATURES_FOR_TWO_HANDS:
                try:
                    two_hand_features = [
                        float(value)
                        for value in raw_features[:FEATURES_FOR_TWO_HANDS]
                    ]
                except (TypeError, ValueError):
                    continue
                if np.isfinite(two_hand_features).all():
                    X_two_hands.append(two_hand_features)
                    y_two_hands.append(label)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

# 2. Validate dataset
counts = Counter(y)

print("\nDATASET SUMMARY")
print("Total samples:", len(X))
print("Gesture classes:", len(counts))

for label, count in sorted(counts.items()):
    print(label, ":", count)

if len(counts) < 2 or min(counts.values()) < 2:
    raise ValueError("Insufficient samples per gesture class.")

# 3. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Train model
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Predictions
test_probabilities = model.predict_proba(X_test)
predictions = model.classes_[np.argmax(test_probabilities, axis=1)]
prediction_confidence = np.max(test_probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)
confidence_sweep = {}
for threshold in (0.35, 0.45, 0.55, 0.65, 0.75):
    accepted = prediction_confidence >= threshold
    confidence_sweep[str(threshold)] = {
        "accepted_samples": int(np.count_nonzero(accepted)),
        "coverage": float(np.mean(accepted)),
        "accuracy": (
            float(accuracy_score(y_test[accepted], predictions[accepted]))
            if np.any(accepted)
            else 0.0
        ),
    }
precision = precision_score(
    y_test, predictions, average="macro", zero_division=0
)
recall = recall_score(
    y_test, predictions, average="macro", zero_division=0
)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, predictions, average="weighted", zero_division=0
)

# 6. Inference latency
start = time.perf_counter()

for sample in X_test:
    model.predict_proba(sample.reshape(1, -1))

elapsed = time.perf_counter() - start

average_latency_ms = (
    elapsed / len(X_test)
) * 1000

# 7. Detailed evaluation
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

two_hand_evaluation = None
if X_two_hands:
    X_two_hands = np.asarray(X_two_hands, dtype=np.float32)
    y_two_hands = np.asarray(y_two_hands)
    dual_counts = Counter(y_two_hands)
    eligible_dual_labels = {
        label for label, count in dual_counts.items() if count >= 5
    }
    dual_mask = np.isin(y_two_hands, list(eligible_dual_labels))
    X_dual = X_two_hands[dual_mask]
    y_dual = y_two_hands[dual_mask]

    if len(eligible_dual_labels) >= 2:
        dual_X_train, dual_X_test, dual_y_train, dual_y_test = train_test_split(
            X_dual,
            y_dual,
            test_size=0.20,
            random_state=42,
            stratify=y_dual,
        )
        dual_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        dual_model.fit(dual_X_train, dual_y_train)
        dual_predictions = dual_model.predict(dual_X_test)
        two_hand_evaluation = {
            "accuracy": float(accuracy_score(dual_y_test, dual_predictions)),
            "test_samples": int(len(dual_y_test)),
            "training_samples": int(len(dual_y_train)),
            "eligible_classes": sorted(eligible_dual_labels),
            "class_counts": dict(dual_counts),
            "classification_report": classification_report(
                dual_y_test,
                dual_predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

print("\n========== GESTUREBRIDGE EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision: {precision * 100:.2f}%")
print(f"Macro Recall: {recall * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
print(f"Average Inference Latency: {average_latency_ms:.3f} ms")
if two_hand_evaluation:
    print(
        "Two-hand held-out accuracy: "
        f"{two_hand_evaluation['accuracy'] * 100:.2f}% "
        f"({two_hand_evaluation['test_samples']} test samples)"
    )

    with open(TWO_HAND_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(two_hand_evaluation, file, indent=4)

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nCONFUSION MATRIX:")
print(matrix)

# 8. Save evaluation report without replacing the deployed model bundle.
os.makedirs("backend/models", exist_ok=True)
results = {
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "average_latency_ms": float(average_latency_ms),
    "test_samples": int(len(X_test)),
    "training_samples": int(len(X_train)),
    "gesture_classes": model.classes_.tolist(),
    "class_counts": dict(counts),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "confidence_threshold_sweep": confidence_sweep,
    "feature_count": int(X.shape[1]),
    "two_hand_model_trained_separately": True,
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("REPORT SAVED:", REPORT_PATH)
print("\nEvaluation completed successfully!")

