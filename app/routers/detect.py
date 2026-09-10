from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2

from services.detection import detect_plates, read_plate

router = APIRouter()


@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload a single image. Returns detected plates with text, bbox, confidence.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections = detect_plates(frame)

    plates = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = frame[y1:y2, x1:x2]

        text, ocr_conf = read_plate(crop)

        plates.append({
            "text": text,
            "bbox": det["bbox"],
            "detection_confidence": det["confidence"],
            "ocr_confidence": ocr_conf
        })

    return {
        "filename": file.filename,
        "total_plates": len(plates),
        "plates": plates
    }