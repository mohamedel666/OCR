from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.ocr.crop_text import crop_image
from backend.ocr.ocr_orchesterator import process_crop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUESTS_DIR = PROJECT_ROOT / "runtime" / "requests"
REQUESTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

app = FastAPI(
    title="OCR API",
    description="Arabic and English OCR with script classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/files",
    StaticFiles(directory=str(REQUESTS_DIR)),
    name="files",
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "OCR API"}


@app.post("/api/ocr")
def run_ocr(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPG, JPEG, or WEBP image.",
        )

    request_id = uuid4().hex
    request_dir = REQUESTS_DIR / request_id
    crops_dir = request_dir / "crops"
    request_dir.mkdir(parents=True)

    original_path = request_dir / f"original{extension}"

    try:
        content = file.file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        original_path.write_bytes(content)

        image = cv2.imread(str(original_path))
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file could not be decoded as an image.",
            )

        crop_paths = crop_image(original_path, crops_dir)
        results = []

        for crop_path in crop_paths:
            prediction, predicted_script, classifier_confidence = (
                process_crop(crop_path)
            )

            results.append({
                "crop": crop_path.name,
                "crop_url": (
                    f"/files/{request_id}/crops/{crop_path.name}"
                ),
                "text": prediction.get("text", ""),
                "confidence": prediction.get("confidence", 0.0),
                "model": prediction.get("model", "unknown"),
                "predicted_script": predicted_script,
                "classifier_confidence": classifier_confidence,
                "needs_review": prediction.get("needs_review", True),
                "reason": prediction.get("reason", ""),
            })

        return {
            "request_id": request_id,
            "original_url": (
                f"/files/{request_id}/original{extension}"
            ),
            "total_crops": len(results),
            "needs_review": any(
                item["needs_review"] for item in results
            ),
            "results": results,
            "message": (
                "No text regions were detected."
                if not results
                else "OCR processing completed."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {exc}",
        ) from exc
    finally:
        file.file.close()
