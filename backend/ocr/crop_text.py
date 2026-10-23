
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


_detector = None


def get_detector():
    global _detector

    if _detector is None:
        _detector = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    return _detector


def order_points(points):
    points = np.array(points, dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def perspective_crop(image, points):
    points = order_points(points)
    tl, tr, br, bl = points

    width = int(max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    ))
    height = int(max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    ))

    if width <= 0 or height <= 0:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(points, destination)

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
    )


def crop_image(image_path, output_dir):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = get_detector().predict(str(image_path))
    crop_paths = []

    for result in results:
        data = result.json

        if isinstance(data, str):
            data = json.loads(data)

        polygons = data.get("res", {}).get("dt_polys", [])

        for polygon in polygons:
            crop = perspective_crop(image, polygon)

            if crop is None:
                continue

            crop_path = output_dir / f"text_{len(crop_paths) + 1:03d}.png"

            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not save crop: {crop_path}")

            crop_paths.append(crop_path)

    return crop_paths
