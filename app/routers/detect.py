from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import os
import uuid
import shutil
import tempfile
from pathlib import Path

from services.detection import detect_plates, read_plate
from services.video_detection import process_video


router = APIRouter()

# ============================================================
# CROPS STORAGE DIRECTORY
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PLATES_DIR = BASE_DIR / "data" / "crops"
PLATES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE ENDPOINT
# ============================================================
@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections = detect_plates(frame)
    plates = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        pad = 12
        y1_crop = max(0, y1 - pad)
        y2_crop = min(frame.shape[0], y2 + pad)
        x1_crop = max(0, x1 - pad)
        x2_crop = min(frame.shape[1], x2 + pad)

        crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        text, ocr_conf = read_plate(crop)

        crop_url = None
        if crop is not None and crop.size > 0:
            crop_filename = f"crop_{uuid.uuid4().hex[:8]}.jpg"
            crop_save_path = PLATES_DIR / crop_filename
            cv2.imwrite(str(crop_save_path), crop)
            crop_url = f"/crops/{crop_filename}"

        plates.append({
            "text": text,
            "plate_text": text,
            "bbox": det["bbox"],
            "detection_confidence": det["confidence"],
            "confidence": det["confidence"],
            "ocr_confidence": ocr_conf,
            "image_url": crop_url,
            "plate_image": crop_url
        })

    return {
        "filename": file.filename,
        "total_plates": len(plates),
        "plates": plates
    }


# ============================================================
# VIDEO ENDPOINT  ✅ NEW
# ============================================================
@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detections = process_video(
            video_path=tmp_path,
            source=file.filename or "Uploaded Video"
        )

        plates = []
        for det in detections:
            plates.append({
                "text": det.get("plate_text", ""),
                "plate_text": det.get("plate_text", ""),
                "track_id": det.get("track_id", "N/A"),
                "detection_confidence": det.get("detection_confidence", 0),
                "confidence": det.get("confidence", 0),
                "ocr_confidence": det.get("confidence", 0),
                "image_url": det.get("crop_image"),
                "plate_image": det.get("crop_image"),
                "source": det.get("source", "Uploaded Video"),
                "timestamp": det.get("timestamp"),
            })

        return {
            "filename": file.filename,
            "total_plates": len(plates),
            "plates": plates
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass