from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import os
import uuid

from services.detection import detect_plates, read_plate
from services.video_detection import process_video


router = APIRouter()


# ============================================================
# POST /detect/image
# ============================================================

@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload a single image.
    Returns detected plates with text, bbox and confidence.
    """

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )

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


# ============================================================
# POST /detect/video
# ============================================================

@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Upload a video and process it using YOLO tracking
    and PaddleOCR.

    Each tracked number plate is OCR processed only once.

    Unique detections are saved into MySQL.
    """

    # --------------------------------------------------------
    # Validate video
    # --------------------------------------------------------

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="File must be a video"
        )

    # --------------------------------------------------------
    # Create uploads directory
    # --------------------------------------------------------

    upload_dir = "uploads"

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate unique temporary filename
    # --------------------------------------------------------

    original_filename = file.filename or "uploaded_video.mp4"

    extension = os.path.splitext(
        original_filename
    )[1]

    if not extension:
        extension = ".mp4"

    temp_filename = f"{uuid.uuid4().hex}{extension}"

    video_path = os.path.join(
        upload_dir,
        temp_filename
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    try:

        contents = await file.read()

        with open(video_path, "wb") as video_file:
            video_file.write(contents)

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        detections = process_video(
            video_path=video_path,
            source=original_filename
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "filename": original_filename,
            "total_detections": len(detections),
            "detections": detections
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print(f"[VIDEO ERROR] {error}")

        raise HTTPException(
            status_code=500,
            detail="Video processing failed"
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded video
        # ----------------------------------------------------

        if os.path.exists(video_path):

            try:
                os.remove(video_path)

            except OSError:
                pass