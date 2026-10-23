from paddleocr import PaddleOCR
import cv2
import os

IMAGE_PATH = "images/image1.png"
OUTPUT_PATH = "images/detected1.png"

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
# Run OCR
result = ocr.predict(IMAGE_PATH)

# Load original image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Draw detected text regions
for res in result:
    data = res.json

    # PaddleOCR 3.x result structure
    if isinstance(data, str):
        import json
        data = json.loads(data)

    for box in data.get("res", {}).get("rec_polys", []):
        points = [(int(x), int(y)) for x, y in box]

        for i in range(len(points)):
            cv2.line(
                image,
                points[i],
                points[(i + 1) % len(points)],
                (0, 255, 0),
                2,
            )

cv2.imwrite(OUTPUT_PATH, image)

print(f"Done: {OUTPUT_PATH}")