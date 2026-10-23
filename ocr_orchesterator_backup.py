
import json
import os
import re
from paddleocr import PaddleOCR

CROPS_DIR = "crops"
OUTPUT_FILE = "ocr_results.json"
ARABIC_THRESHOLD = 0.80

ARABIC_MODEL_NAME = "arabic_PP-OCRv5_mobile_rec"
ENGLISH_MODEL_NAME = "PP-OCRv6_medium_rec"

print("Loading Arabic OCR...")
arabic_ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name=ARABIC_MODEL_NAME,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# Load English OCR only when needed.
english_ocr = None


def get_english_ocr():
    global english_ocr

    if english_ocr is None:
        print("Loading English OCR...")
        english_ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name=ENGLISH_MODEL_NAME,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    return english_ocr


def extract_prediction(ocr, image_path):
    results = ocr.predict(image_path)
    predictions = []

    for result in results:
        data = result.json

        if isinstance(data, str):
            data = json.loads(data)

        res_data = data.get("res", {})
        texts = res_data.get("rec_texts", [])
        scores = res_data.get("rec_scores", [])

        for text, score in zip(texts, scores):
            text = str(text).strip()
            if text:
                predictions.append((text, float(score)))

    if not predictions:
        return {"text": "", "confidence": 0.0}

    return {
        "text": " ".join(text for text, _ in predictions),
        "confidence": sum(score for _, score in predictions)
        / len(predictions),
    }


def script_info(text):
    arabic_count = len(
        re.findall(r"[\u0600-\u06FF]", text)
    )
    latin_count = len(
        re.findall(r"[A-Za-z]", text)
    )

    if arabic_count and latin_count:
        script = "mixed"
    elif arabic_count:
        script = "arabic"
    elif latin_count:
        script = "latin"
    else:
        script = "unknown"

    return {
        "script": script,
        "arabic_chars": arabic_count,
        "latin_chars": latin_count,
    }

def normalize_for_comparison(text):
    """Normalize spacing for comparison without changing the OCR output."""
    text = text.strip()
    text = re.sub(r"\s*([/.,:;()-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def candidate_score(prediction, model_type):
    text = prediction["text"]
    confidence = prediction["confidence"]
    info = script_info(text)

    if not text:
        return -1.0

    score = confidence

    if info["script"] == "mixed":
        score += 0.10
    elif model_type == "arabic" and info["script"] == "arabic":
        score += 0.08
    elif model_type == "english" and info["script"] == "latin":
        score += 0.08
    elif info["script"] == "unknown":
        score -= 0.10

    return score




def is_numeric_like(text):
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    compact = re.sub(r"[\s.,:/()-]", "", text)

    if not compact:
        return False

    digits = sum(char.isdigit() for char in compact)
    letters = sum(char.isalpha() for char in compact)

    return digits >= 2 and digits / len(compact) >= 0.8


def choose_prediction(arabic, english):
    arabic_text = arabic["text"].strip()
    english_text = english["text"].strip()

    arabic_score = candidate_score(arabic, "arabic")
    english_score = candidate_score(english, "english")

    if not arabic_text and not english_text:
        return {
            "text": "",
            "confidence": 0.0,
            "model": "none",
            "reason": "Both models returned empty text",
            "needs_review": True,
        }

    if not arabic_text:
        selected, model = english, ENGLISH_MODEL_NAME
        reason = "Arabic model returned no text"

    elif not english_text:
        selected, model = arabic, ARABIC_MODEL_NAME
        reason = "English model returned no text"

    else:
        arabic_script = script_info(arabic_text)
        english_script = script_info(english_text)

        arabic_numeric = is_numeric_like(arabic_text)
        english_numeric = is_numeric_like(english_text)

        # Prefer a clear Arabic-script result over Latin gibberish.
        if arabic_script in ("arabic", "mixed") and english_script == "latin":
            selected, model = arabic, ARABIC_MODEL_NAME
            reason = "Arabic-script result preferred"

        # Prefer a clear Latin-script result over Arabic gibberish.
        elif english_script in ("latin", "mixed") and arabic_script == "arabic":
            selected, model = english, ENGLISH_MODEL_NAME
            reason = "Latin-script result preferred"

        # Prefer a numeric-looking result over alphabetic gibberish.
        elif arabic_numeric and not english_numeric:
            selected, model = arabic, ARABIC_MODEL_NAME
            reason = "Arabic candidate looks numeric"

        elif english_numeric and not arabic_numeric:
            selected, model = english, ENGLISH_MODEL_NAME
            reason = "English candidate looks numeric"

        # Otherwise, fall back to the existing candidate scores.
        elif arabic_score >= english_score:
            selected, model = arabic, ARABIC_MODEL_NAME
            reason = "Arabic candidate had the higher adjusted score"

        else:
            selected, model = english, ENGLISH_MODEL_NAME
            reason = "English candidate had the higher adjusted score"

    needs_review = False

    if arabic_text and english_text:
        texts_match = (
            normalize_for_comparison(arabic_text)
            == normalize_for_comparison(english_text)
        )

        if not texts_match:
            needs_review = True
            reason += "; model outputs differ"

        elif selected["confidence"] < 0.85:
            needs_review = True
            reason += "; selected confidence is low"

        else:
            reason += "; models agree after normalization"

    else:
        needs_review = True

    return {
        "text": selected["text"],
        "confidence": selected["confidence"],
        "model": model,
        "reason": reason,
        "needs_review": needs_review,
    }



def process_crop(image_path):
    # First pass: Arabic recognizer acts as a probe.
    arabic = extract_prediction(arabic_ocr, image_path)
    info = script_info(arabic["text"])

    # Clear Arabic result: skip the English recognizer.
    if (
        info["script"] == "arabic"
        and arabic["confidence"] >= ARABIC_THRESHOLD
    ):
        selected = {
            "text": arabic["text"],
            "confidence": arabic["confidence"],
            "model": ARABIC_MODEL_NAME,
            "reason": (
                "Arabic probe detected Arabic text with sufficient "
                "confidence; English model skipped"
            ),
            "needs_review": arabic["confidence"] < 0.85,
        }

        english = {
            "text": "",
            "confidence": 0.0,
            "status": "not_run",
        }

        return selected, arabic, english

    # Latin, mixed, numeric-only, empty, or uncertain:
    # run English OCR and compare candidates.
    english = extract_prediction(
        get_english_ocr(),
        image_path,
    )

    selected = choose_prediction(arabic, english)
    selected["reason"] = (
        f"Both models compared. {selected['reason']}"
    )

    return selected, arabic, english


def main():
    if not os.path.isdir(CROPS_DIR):
        raise FileNotFoundError(
            f"Crop directory not found: {CROPS_DIR}"
        )

    files = sorted(
        filename
        for filename in os.listdir(CROPS_DIR)
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No cropped images found in {CROPS_DIR}"
        )

    report = []

    for filename in files:
        image_path = os.path.join(CROPS_DIR, filename)

        selected, arabic, english = process_crop(image_path)

        item = {
            "image": filename,
            **selected,
            "probe_script": script_info(arabic["text"])["script"],
            "candidates": {
                "arabic": arabic,
                "english": english,
            },
        }
        report.append(item)

        print("\n" + "=" * 60)
        print(f"Crop         : {filename}")
        print(f"Probe script : {item['probe_script']}")
        print(f"Chosen text  : {selected['text']}")
        print(f"Confidence   : {selected['confidence']:.4f}")
        print(f"Chosen model : {selected['model']}")
        print(f"Reason       : {selected['reason']}")
        print(f"Needs review : {selected['needs_review']}")
        print(f"Arabic text  : {arabic['text']}")
        print(f"English text : {english['text'] or english.get('status', '')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nFinished. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
