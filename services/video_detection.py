import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime

from app.database import save_detection


# ============================================
# LOAD YOLO MODEL
# ============================================

model = YOLO("models/best.pt")
names = model.names


# ============================================
# LOAD PADDLE OCR
# ============================================

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)


# ============================================
# PROCESS VIDEO
# ============================================

def process_video(video_path: str, source: str):
    """
    Process uploaded video using YOLO tracking
    and PaddleOCR.

    Each tracked number plate is OCR processed
    only once and saved to MySQL.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    # Track ID -> detected plate
    id_to_plate = {}

    # Track IDs already saved to database
    saved_ids = set()

    # Results returned by this function
    detections = []

    frame_count = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process every 3rd frame
            if frame_count % 3 != 0:
                continue

            # Resize frame
            frame = cv2.resize(frame, (1020, 600))

            # YOLO tracking
            results = model.track(
                frame,
                persist=True,
                verbose=False
            )

            # No tracking IDs
            if results[0].boxes.id is None:
                continue

            # Get tracking IDs
            ids = (
                results[0]
                .boxes
                .id
                .cpu()
                .numpy()
                .astype(int)
            )

            # Get bounding boxes
            boxes = (
                results[0]
                .boxes
                .xyxy
                .cpu()
                .numpy()
                .astype(int)
            )

            # Get class IDs
            class_ids = (
                results[0]
                .boxes
                .cls
                .int()
                .cpu()
                .tolist()
            )

            # Get detection confidence
            confidences = (
                results[0]
                .boxes
                .conf
                .cpu()
                .numpy()
            )

            # Process every tracked object
            for track_id, box, class_id, confidence in zip(
                ids,
                boxes,
                class_ids,
                confidences
            ):

                x1, y1, x2, y2 = box

                label = names[class_id]

                # Only process number plates
                if label.lower() != "numberplate":
                    continue

                # Keep bounding box inside frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Crop number plate
                cropped_plate = frame[y1:y2, x1:x2]

                if cropped_plate.size == 0:
                    continue

                # ========================================
                # OCR ONLY ONCE PER TRACK ID
                # ========================================

                if track_id not in id_to_plate:

                    result = ocr.predict(cropped_plate)

                    plate_text = ""

                    if (
                        result
                        and isinstance(result, list)
                        and "rec_texts" in result[0]
                    ):

                        rec_texts = result[0]["rec_texts"]

                        plate_text = " ".join(
                            rec_texts
                        ).strip()

                    # If OCR successfully detected text
                    if plate_text:

                        id_to_plate[track_id] = plate_text

                        # =================================
                        # SAVE TO MYSQL ONLY ONCE
                        # =================================

                        if track_id not in saved_ids:

                            timestamp = datetime.now()

                            success = save_detection(
                                plate_text=plate_text,
                                track_id=int(track_id),
                                confidence=float(confidence),
                                timestamp=timestamp,
                                source=source
                            )

                            if success:

                                saved_ids.add(track_id)

                                detection = {
                                    "plate_text": plate_text,
                                    "track_id": int(track_id),
                                    "confidence": float(confidence),
                                    "timestamp": timestamp.isoformat(),
                                    "source": source
                                }

                                detections.append(detection)

                                print(
                                    f"[INFO] Saved: "
                                    f"ID={track_id}, "
                                    f"Plate={plate_text}, "
                                    f"Confidence={float(confidence):.4f}"
                                )

    finally:

        cap.release()

    return detections