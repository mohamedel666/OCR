from paddleocr import PaddleOCR
import os


CROPS_DIR = "crops"


ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


files = sorted(
    [
        f
        for f in os.listdir(CROPS_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)


if not files:
    raise FileNotFoundError("No cropped images found.")


for filename in files:

    image_path = os.path.join(CROPS_DIR, filename)

    result = ocr.predict(image_path)

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    for res in result:

        data = res.json

        if isinstance(data, str):
            import json
            data = json.loads(data)

        res_data = data.get("res", {})

        texts = res_data.get("rec_texts", [])
        scores = res_data.get("rec_scores", [])

        for text, score in zip(texts, scores):
            print(f"Text       : {text}")
            print(f"Confidence : {score:.4f}")