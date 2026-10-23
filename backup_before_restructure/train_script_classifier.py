
import io
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = Path("datasets")
MODEL_PATH = Path("script_classifier.joblib")

PARQUET_FILES = [
    DATA_DIR / "test_ar.parquet",
    DATA_DIR / "test_en.parquet",
    DATA_DIR / "test_numbers.parquet",
]

IMAGE_SIZE = (64, 128)  # width, height

hog = cv2.HOGDescriptor(
    (64, 128),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)


def get_label(text):
    text = str(text).strip()

    arabic = any("\u0600" <= c <= "\u06FF" for c in text)
    latin = any(c.isascii() and c.isalpha() for c in text)

    compact = "".join(c for c in text if not c.isspace())

    digits = sum(c.isdigit() for c in compact)
    numeric_ratio = digits / max(len(compact), 1)

    if arabic and latin:
        return "mixed"
    if arabic:
        return "arabic"
    if latin:
        return "latin"
    if digits and numeric_ratio >= 0.8:
        return "numeric"

    return None


def extract_features(image_bytes):
    image = Image.open(io.BytesIO(bytes(image_bytes))).convert("L")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)
    features = hog.compute(image_array)

    return features.flatten()


def main():
    features = []
    labels = []

    for parquet_path in PARQUET_FILES:
        if not parquet_path.exists():
            print(f"Skipping missing file: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        for row in df.itertuples(index=False):
            label = get_label(row.full_text)

            if label is None:
                continue

            try:
                vector = extract_features(row.image_bytes)
                features.append(vector)
                labels.append(label)
            except Exception as exc:
                print(f"Skipped an unreadable image: {exc}")

    if not features:
        raise RuntimeError("No usable training images were found.")

    X = np.asarray(features)
    y = np.asarray(labels)

    print("Samples:", len(y))
    print("Classes:", dict(zip(*np.unique(y, return_counts=True))))

    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise RuntimeError(
            "Need at least two classes and at least two samples per class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved classifier to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
