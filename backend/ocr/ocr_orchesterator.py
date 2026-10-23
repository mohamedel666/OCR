
import json
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
from paddleocr import PaddleOCR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CROPS_DIR = PROJECT_ROOT / "crops"
OUTPUT_FILE = PROJECT_ROOT / "ocr_results.json"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "script_classifier.joblib"

ARABIC_MODEL_NAME = "arabic_PP-OCRv5_mobile_rec"
ENGLISH_MODEL_NAME = "PP-OCRv6_medium_rec"

CLASSIFIER_THRESHOLD = 0.65
REVIEW_THRESHOLD = 0.85

hog = cv2.HOGDescriptor(
    (64, 128),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(
        "Train the visual classifier first by running "
        "train_script_classifier.py"
    )

script_classifier = joblib.load(CLASSIFIER_PATH)


def make_ocr(model_name):
    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name=model_name,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


print("Loading Arabic OCR...")
arabic_ocr = make_ocr(ARABIC_MODEL_NAME)

english_ocr = None


def get_english_ocr():
    global english_ocr

    if english_ocr is None:
        print("Loading English OCR...")
        english_ocr = make_ocr(ENGLISH_MODEL_NAME)

    return english_ocr


def classify_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, (64, 128))
    features = hog.compute(image).reshape(1, -1)

    probabilities = script_classifier.predict_proba(features)[0]
    classes = script_classifier.named_steps[
        "classifier"
    ].classes_

    index = int(np.argmax(probabilities))

    return str(classes[index]), float(probabilities[index])


def extract_prediction(ocr, image_path):
    results = ocr.predict(str(image_path))
    texts = []
    scores = []

    for result in results:
        data = result.json

        if isinstance(data, str):
            data = json.loads(data)

        res = data.get("res", {})
        texts.extend(res.get("rec_texts", []))
        scores.extend(res.get("rec_scores", []))

    pairs = [
        (str(text).strip(), float(score))
        for text, score in zip(texts, scores)
        if str(text).strip()
    ]

    if not pairs:
        return {"text": "", "confidence": 0.0}

    return {
        "text": " ".join(text for text, _ in pairs),
        "confidence": sum(score for _, score in pairs) / len(pairs),
    }


def script_info(text):
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    if arabic and latin:
        script = "mixed"
    elif arabic:
        script = "arabic"
    elif latin:
        script = "latin"
    else:
        script = "unknown"

    return script


def normalize(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def candidate_score(prediction, model_type):
    text = prediction["text"].strip()

    if not text:
        return -1.0

    score = prediction["confidence"]
    script = script_info(text)

    if model_type == "arabic" and script == "arabic":
        score += 0.02
    elif model_type == "english" and script == "latin":
        score += 0.02
    elif script == "unknown":
        score -= 0.02

    return score


def choose_prediction(arabic, english):
    if not arabic["text"] and not english["text"]:
        return {
            "text": "",
            "confidence": 0.0,
            "model": "none",
            "reason": "Both models returned empty text",
            "needs_review": True,
        }

    if not arabic["text"]:
        selected, model = english, ENGLISH_MODEL_NAME
        reason = "Arabic OCR returned no text"

    elif not english["text"]:
        selected, model = arabic, ARABIC_MODEL_NAME
        reason = "English OCR returned no text"

    elif candidate_score(arabic, "arabic") >= candidate_score(
        english, "english"
    ):
        selected, model = arabic, ARABIC_MODEL_NAME
        reason = "Arabic candidate had the higher adjusted score"

    else:
        selected, model = english, ENGLISH_MODEL_NAME
        reason = "English candidate had the higher adjusted score"

    disagree = (
        bool(arabic["text"])
        and bool(english["text"])
        and normalize(arabic["text"]) != normalize(english["text"])
    )

    needs_review = (
        disagree
        or selected["confidence"] < REVIEW_THRESHOLD
        or not arabic["text"]
        or not english["text"]
    )

    if disagree:
        reason += "; model outputs differ"

    return {
        "text": selected["text"],
        "confidence": selected["confidence"],
        "model": model,
        "reason": reason,
        "needs_review": needs_review,
    }





def process_crop(image_path):
    # 1. Classify the image before running OCR.
    predicted_script, classifier_confidence = classify_image(image_path)

    # 2. Route clear Arabic/Latin predictions to the relevant model.
    if classifier_confidence >= CLASSIFIER_THRESHOLD:
        if predicted_script == "arabic":
            first_model = "arabic"
        elif predicted_script == "latin":
            first_model = "english"
        else:
            first_model = "both"
    else:
        first_model = "both"

    comparison_performed = False
    fallback_used = False

    # 3. Compare both models for uncertain or numeric predictions.
    if first_model == "both":
        arabic = extract_prediction(arabic_ocr, image_path)
        english = extract_prediction(get_english_ocr(), image_path)

        result = choose_prediction(arabic, english)
        comparison_performed = True

        result["reason"] = (
            f"Classifier predicted {predicted_script} "
            f"({classifier_confidence:.3f}); both OCR models compared. "
            + result["reason"]
        )

    else:
        if first_model == "arabic":
            first = extract_prediction(arabic_ocr, image_path)
            first_name = ARABIC_MODEL_NAME
            second_model = get_english_ocr()
            second_name = ENGLISH_MODEL_NAME
        else:
            first = extract_prediction(get_english_ocr(), image_path)
            first_name = ENGLISH_MODEL_NAME
            second_model = arabic_ocr
            second_name = ARABIC_MODEL_NAME

        # 4. Use the second model if the first fails or has low confidence.
        if (
            not first["text"].strip()
            or first["confidence"] < REVIEW_THRESHOLD
        ):
            second = extract_prediction(second_model, image_path)
            fallback_used = True
            comparison_performed = True

            if first_model == "arabic":
                arabic, english = first, second
            else:
                arabic, english = second, first

            result = choose_prediction(arabic, english)
            result["reason"] = (
                f"{first_name} failed or had low confidence; "
                f"{second_name} fallback attempted. "
                + result["reason"]
            )
        else:
            result = {
                "text": first["text"],
                "confidence": first["confidence"],
                "model": first_name,
                "reason": f"Visual classifier routed to {first_name}",
            }

    # 5. Apply review rules consistently.
    text = result.get("text", "").strip()
    confidence = result.get("confidence", 0.0)
    detected_script = script_info(text)
    review_reasons = []

    if not text:
        review_reasons.append("OCR returned empty text")

    if confidence < REVIEW_THRESHOLD:
        review_reasons.append("OCR confidence below review threshold")

    if detected_script == "mixed":
        review_reasons.append("Mixed-script output")

    if fallback_used:
        review_reasons.append("Fallback OCR was required")

    if (
        predicted_script in ("numeric", "unknown")
        and not comparison_performed
    ):
        review_reasons.append(
            "Numeric/unknown classification without model comparison"
        )

    result["needs_review"] = bool(review_reasons)

    if review_reasons:
        result["reason"] += "; Review: " + "; ".join(review_reasons)

    result["predicted_script"] = predicted_script
    result["classifier_confidence"] = classifier_confidence
    result["comparison_performed"] = comparison_performed
    result["fallback_used"] = fallback_used

    return result, predicted_script, classifier_confidence
    # 1. Classify the image before OCR.
    predicted_script, classifier_confidence = classify_image(image_path)

    # 2. Choose the first model using the visual classifier.
    if classifier_confidence >= CLASSIFIER_THRESHOLD:
        if predicted_script == "arabic":
            first_model = "arabic"
        elif predicted_script == "latin":
            first_model = "english"
        else:
            first_model = "both"
    else:
        first_model = "both"

    # 3. Run both models for uncertain, numeric, or mixed cases.
    if first_model == "both":
        arabic = extract_prediction(arabic_ocr, image_path)
        english = extract_prediction(get_english_ocr(), image_path)
        result = choose_prediction(arabic, english)

        result["reason"] = (
            f"Classifier predicted {predicted_script} "
            f"({classifier_confidence:.3f}); both OCR models compared. "
            + result["reason"]
        )

    else:
        # Run the preferred model first.
        if first_model == "arabic":
            first = extract_prediction(arabic_ocr, image_path)
            first_name = ARABIC_MODEL_NAME
            second_model = get_english_ocr()
            second_name = ENGLISH_MODEL_NAME
        else:
            first = extract_prediction(get_english_ocr(), image_path)
            first_name = ENGLISH_MODEL_NAME
            second_model = arabic_ocr
            second_name = ARABIC_MODEL_NAME

        # 4. If the first model fails or has low confidence,
        # run the fallback model and compare both candidates.
        if (
            not first["text"].strip()
            or first["confidence"] < REVIEW_THRESHOLD
        ):
            second = extract_prediction(second_model, image_path)

            if first_model == "arabic":
                arabic, english = first, second
            else:
                arabic, english = second, first

            result = choose_prediction(arabic, english)
            result["reason"] = (
                f"{first_name} failed or had low confidence; "
                f"{second_name} fallback attempted. "
                + result["reason"]
            )
        else:
            result = {
                "text": first["text"],
                "confidence": first["confidence"],
                "model": first_name,
                "reason": f"Visual classifier routed to {first_name}",
                "needs_review": False,
            }

    # 5. Determine whether the result needs human review.
    text = result.get("text", "").strip()
    confidence = result.get("confidence", 0.0)
    detected_script = script_info(text)

    review_reasons = []

    if not text:
        review_reasons.append("OCR returned empty text")

    if confidence < REVIEW_THRESHOLD:
        review_reasons.append("OCR confidence below review threshold")

    if detected_script == "mixed":
        review_reasons.append("Mixed-script output")

    if predicted_script in ("numeric", "unknown"):
        review_reasons.append(
            f"Classifier predicted {predicted_script}; fallback comparison used"
        )

    if "fallback attempted" in result.get("reason", "").lower():
        review_reasons.append("Fallback OCR was required")

    result["needs_review"] = bool(review_reasons)

    if review_reasons:
        result["reason"] += "; Review: " + "; ".join(review_reasons)

    result["predicted_script"] = predicted_script
    result["classifier_confidence"] = classifier_confidence

    return result, predicted_script, classifier_confidence

def main():
    if not CROPS_DIR.is_dir():
        raise FileNotFoundError(f"Crop directory not found: {CROPS_DIR}")

    files = sorted(
        p for p in CROPS_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not files:
        raise FileNotFoundError(f"No crop images found in {CROPS_DIR}")

    report = []

    for image_path in files:
        selected, predicted_script, classifier_confidence = process_crop(
            image_path
        )

        item = {
            "image": image_path.name,
            "predicted_script": predicted_script,
            "classifier_confidence": classifier_confidence,
            **selected,
        }
        report.append(item)

        print("\n" + "=" * 50)
        print("Crop:", image_path.name)
        print("Visual classification:", predicted_script)
        print(f"Classifier confidence: {classifier_confidence:.3f}")
        print("Text:", selected["text"])
        print("OCR model:", selected["model"])
        print(f"OCR confidence: {selected['confidence']:.3f}")
        print("Reason:", selected["reason"])
        print("Needs review:", selected["needs_review"])

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
