
import csv
import os
import json
import time
import joblib
import numpy as np
from gesture_features import FEATURES_PER_HAND

from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

DATA_PATH = "backend/data/independent_test_samples.csv"
MODEL_PATH = "backend/models/gesture_model.pkl"
REPORT_PATH = "backend/models/independent_evaluation_report.json"

# Check files
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Test dataset not found: {DATA_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

# Load the existing trained model
saved = joblib.load(MODEL_PATH)
model = saved["model"]
classes = list(saved["classes"])

# Load independent test samples
X = []
y = []

with open(DATA_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    next(reader, None)

    for row in reader:
        if not row:
            continue
        label = row[0].strip().lower()
        if not label:
            continue

        if label not in classes:
            raise ValueError(
                f"Unknown gesture label in test dataset: {label}"
            )

        raw_features = row[1:]
        if len(raw_features) < FEATURES_PER_HAND:
            continue

        try:
            features = [float(value) for value in raw_features[:FEATURES_PER_HAND]]
        except (TypeError, ValueError):
            continue
        if len(features) != saved["feature_count"]:
            continue

        if not np.isfinite(features).all():
            raise ValueError("Invalid numeric feature detected.")

        X.append(features)
        y.append(label)

if not X:
    raise ValueError("No independent test samples found.")

X = np.asarray(X, dtype=float)
y = np.asarray(y)

# Check which gestures were recorded
counts = Counter(y)
evaluation_labels = sorted(counts)

print("\nINDEPENDENT TEST DATASET")
print("Total samples:", len(y))
print("Gesture classes recorded:", len(counts))

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")

# Predict using the saved model
predictions = []

latencies = []

for sample in X:
    start = time.perf_counter()
    prediction = model.predict(sample.reshape(1, -1))[0]
    end = time.perf_counter()

    predictions.append(str(prediction))
    latencies.append((end - start) * 1000)

predictions = np.asarray(predictions)

# Calculate metrics
accuracy = accuracy_score(y, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    y,
    predictions,
    labels=evaluation_labels,
    average="macro",
    zero_division=0
)

weighted_f1 = precision_recall_fscore_support(
    y,
    predictions,
    labels=classes,
    average="weighted",
    zero_division=0
)[2]

report = classification_report(
    y,
    predictions,
    labels=evaluation_labels,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y,
    predictions,
    labels=evaluation_labels
)

# Prepare evaluation report
results = {
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "weighted_f1": float(weighted_f1),
    "average_inference_latency_ms": float(np.mean(latencies)),
    "total_test_samples": int(len(y)),
    "gesture_classes_recorded": int(len(counts)),
    "sample_counts": dict(counts),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "class_order": evaluation_labels,
    "model_classes": classes
}

# Save report separately
with open(REPORT_PATH, "w") as f:
    json.dump(results, f, indent=4)

# Display results
print("\n========== INDEPENDENT EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision: {precision * 100:.2f}%")
print(f"Macro Recall: {recall * 100:.2f}%")
print(f"Macro F1: {f1 * 100:.2f}%")
print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
print(f"Average Inference Latency: {np.mean(latencies):.3f} ms")

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y,
        predictions,
        labels=evaluation_labels,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX:")
print(matrix)

print("\nReport saved to:", REPORT_PATH)
