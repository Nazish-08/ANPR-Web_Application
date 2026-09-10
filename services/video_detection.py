import cv2
import re
from collections import Counter, defaultdict
from datetime import datetime

from ultralytics import YOLO
from paddleocr import PaddleOCR

from app.database import save_detection


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("[INFO] Loading YOLO model...")

model = YOLO("models/best.pt")
names = model.names


# ============================================================
# LOAD PADDLE OCR
# ============================================================

print("[INFO] Loading PaddleOCR...")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

print("[INFO] Models loaded successfully.")


# ============================================================
# OCR TEXT CLEANING
# ============================================================

def clean_plate_text(text: str) -> str:
    """
    Clean OCR text.

    Keeps only uppercase English letters and numbers.
    Spaces and special characters are removed.
    """

    if not text:
        return ""

    text = str(text).upper().strip()

    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# INDIAN NUMBER PLATE VALIDATION
# ============================================================

def is_valid_plate(text: str) -> bool:
    """
    Validate a general Indian vehicle registration format.

    Examples accepted:

        MH12AB1234
        TS07JS9670
        UP32AB1234
        DL01AB1234
        KA01MN1234
        GJ01AB1234
        RJ14CD5678

    State code is NOT hardcoded.

    The function rejects obvious OCR garbage such as:

        A
        1
        SAIDO
        GAIDCO
        CSAIDCO
        9670
    """

    if not text:
        return False

    text = clean_plate_text(text)

    # Indian plates are normally between 8 and 12 characters
    if len(text) < 8 or len(text) > 12:
        return False

    # Must contain both letters and numbers
    if not re.search(r"[A-Z]", text):
        return False

    if not re.search(r"[0-9]", text):
        return False

    # --------------------------------------------------------
    # Normal Indian registration format
    #
    # 2 letters
    # 1 or 2 digits
    # 1 to 3 letters
    # 1 to 4 digits
    # --------------------------------------------------------

    normal_pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"

    if re.fullmatch(normal_pattern, text):
        return True

    # --------------------------------------------------------
    # Bharat Series format
    #
    # Example:
    # BH12AB1234
    # --------------------------------------------------------

    bharat_pattern = r"^BH[0-9]{2}[A-Z]{1,2}[0-9]{4}$"

    if re.fullmatch(bharat_pattern, text):
        return True

    return False


# ============================================================
# OCR RESULT EXTRACTION
# ============================================================

def extract_ocr_text(result) -> str:
    """
    Extract recognition text from PaddleOCR result.

    Returns a cleaned string.
    """

    if not result:
        return ""

    try:

        # PaddleOCR current result format
        if isinstance(result, list):

            for item in result:

                if isinstance(item, dict):

                    rec_texts = item.get("rec_texts")

                    if rec_texts:

                        text = " ".join(
                            str(value)
                            for value in rec_texts
                        )

                        return clean_plate_text(text)

                else:

                    # Some PaddleOCR versions return objects
                    if hasattr(item, "json"):

                        try:
                            data = item.json

                            if isinstance(data, dict):

                                rec_texts = data.get(
                                    "rec_texts",
                                    []
                                )

                                if rec_texts:

                                    text = " ".join(
                                        str(value)
                                        for value in rec_texts
                                    )

                                    return clean_plate_text(
                                        text
                                    )

                        except Exception:
                            pass

        # Direct dictionary
        if isinstance(result, dict):

            rec_texts = result.get("rec_texts")

            if rec_texts:

                text = " ".join(
                    str(value)
                    for value in rec_texts
                )

                return clean_plate_text(text)

    except Exception as error:

        print(
            f"[WARNING] OCR extraction error: {error}"
        )

    return ""


# ============================================================
# PLATE QUALITY SCORE
# ============================================================

def plate_score(text: str) -> int:
    """
    Give a quality score to a candidate plate.

    Higher score means better structural match.
    """

    if not text:
        return 0

    score = 0

    # Strongest signal: complete Indian plate format
    if re.fullmatch(
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$",
        text
    ):
        score += 100

    # Bharat Series
    if re.fullmatch(
        r"^BH[0-9]{2}[A-Z]{1,2}[0-9]{4}$",
        text
    ):
        score += 100

    # Prefer longer complete results
    score += min(len(text) * 2, 24)

    # A valid plate should contain both letters and numbers
    if re.search(r"[A-Z]", text):
        score += 10

    if re.search(r"[0-9]", text):
        score += 10

    return score


# ============================================================
# FIND BEST OCR RESULT
# ============================================================

def choose_best_plate(candidates):
    """
    Choose the best plate from multiple OCR results.

    Voting is used first.

    If multiple candidates exist, the candidate with
    the strongest combination of frequency and format
    quality is selected.
    """

    if not candidates:
        return None

    # Clean all candidates
    cleaned = []

    for candidate in candidates:

        candidate = clean_plate_text(candidate)

        if not candidate:
            continue

        if not is_valid_plate(candidate):
            continue

        cleaned.append(candidate)

    if not cleaned:
        return None

    # Count OCR votes
    counts = Counter(cleaned)

    # Select using frequency + structural quality
    ranked = sorted(
        counts.items(),
        key=lambda item: (
            item[1],
            plate_score(item[0]),
            len(item[0])
        ),
        reverse=True
    )

    return ranked[0][0]


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path: str, source: str):
    """
    Process uploaded video using YOLO tracking and PaddleOCR.

    Optimized behaviour:

    1. Approximately one frame per second is processed.
    2. YOLO detects number plates.
    3. OCR results are collected instead of immediately
       saving the first result.
    4. Invalid OCR results are ignored.
    5. Multiple OCR results are voted on.
    6. The best valid plate is selected.
    7. Duplicate plates are saved only once.
    8. Final detections are inserted into MySQL.
    """

    print(
        f"[INFO] Processing video: {source}"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            "Could not open video file."
        )

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    print(
        f"[INFO] Video FPS: {fps:.2f}"
    )

    print(
        f"[INFO] Video duration: "
        f"{duration:.2f} seconds"
    )

    # ========================================================
    # 1 FRAME PER SECOND
    # ========================================================

    frame_interval = max(
        1,
        int(round(fps))
    )

    # ========================================================
    # OCR STORAGE
    # ========================================================

    # Track ID -> OCR candidates
    track_candidates = defaultdict(list)

    # Track ID -> highest YOLO confidence
    track_confidence = {}

    # Track ID -> latest bounding box
    track_boxes = {}

    # ========================================================
    # GLOBAL OCR CANDIDATES
    # ========================================================

    all_valid_candidates = []

    frame_count = 0
    processed_frames = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # ------------------------------------------------
            # Process approximately one frame every second
            # ------------------------------------------------

            if frame_count % frame_interval != 0:
                continue

            processed_frames += 1

            print(
                f"[INFO] Processing frame "
                f"{frame_count}/{total_frames}"
            )

            # ------------------------------------------------
            # Resize frame
            # ------------------------------------------------

            frame = cv2.resize(
                frame,
                (1020, 600)
            )

            # ------------------------------------------------
            # YOLO tracking
            # ------------------------------------------------

            results = model.track(
                frame,
                persist=True,
                verbose=False
            )

            if not results:
                continue

            result = results[0]

            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue

            # ------------------------------------------------
            # Get tracking IDs
            # ------------------------------------------------

            ids = (
                result
                .boxes
                .id
                .cpu()
                .numpy()
                .astype(int)
            )

            # ------------------------------------------------
            # Get bounding boxes
            # ------------------------------------------------

            boxes = (
                result
                .boxes
                .xyxy
                .cpu()
                .numpy()
                .astype(int)
            )

            # ------------------------------------------------
            # Get class IDs
            # ------------------------------------------------

            class_ids = (
                result
                .boxes
                .cls
                .int()
                .cpu()
                .tolist()
            )

            # ------------------------------------------------
            # Get YOLO confidence
            # ------------------------------------------------

            confidences = (
                result
                .boxes
                .conf
                .cpu()
                .numpy()
            )

            # =================================================
            # PROCESS DETECTED OBJECTS
            # =================================================

            for (
                track_id,
                box,
                class_id,
                confidence
            ) in zip(
                ids,
                boxes,
                class_ids,
                confidences
            ):

                label = names[class_id]

                # ------------------------------------------------
                # Only number plates
                # ------------------------------------------------

                if label.lower() != "numberplate":
                    continue

                confidence = float(
                    confidence
                )

                # ------------------------------------------------
                # YOLO confidence threshold
                # ------------------------------------------------

                if confidence < 0.50:
                    continue

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                x1, y1, x2, y2 = box

                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                # ------------------------------------------------
                # Crop number plate
                # ------------------------------------------------

                cropped_plate = frame[
                    y1:y2,
                    x1:x2
                ]

                if cropped_plate.size == 0:
                    continue

                # ------------------------------------------------
                # Save highest YOLO confidence for track
                # ------------------------------------------------

                previous_confidence = track_confidence.get(
                    int(track_id),
                    0.0
                )

                if confidence > previous_confidence:

                    track_confidence[
                        int(track_id)
                    ] = confidence

                    track_boxes[
                        int(track_id)
                    ] = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                # =================================================
                # OCR
                # =================================================

                try:

                    result_ocr = ocr.predict(
                        cropped_plate
                    )

                    plate_text = extract_ocr_text(
                        result_ocr
                    )

                except Exception as error:

                    print(
                        f"[WARNING] OCR failed "
                        f"for Track ID={track_id}: "
                        f"{error}"
                    )

                    continue

                # =================================================
                # INVALID OCR
                # =================================================

                if not plate_text:

                    print(
                        f"[INFO] Ignored empty OCR "
                        f"| Track ID={track_id}"
                    )

                    continue

                # =================================================
                # VALIDATE OCR
                # =================================================

                if not is_valid_plate(
                    plate_text
                ):

                    print(
                        f"[INFO] Ignored invalid OCR: "
                        f"'{plate_text}' "
                        f"| Track ID={track_id}"
                    )

                    continue

                # =================================================
                # VALID OCR CANDIDATE
                # =================================================

                track_candidates[
                    int(track_id)
                ].append(
                    plate_text
                )

                all_valid_candidates.append(
                    plate_text
                )

                print(
                    f"[INFO] Valid OCR candidate: "
                    f"{plate_text} "
                    f"| Track ID={track_id}"
                )

    finally:

        cap.release()

    # ============================================================
    # VIDEO PROCESSING COMPLETE
    # ============================================================

    print(
        f"[INFO] Video processing completed."
    )

    print(
        f"[INFO] Frames processed: "
        f"{processed_frames}"
    )

    # ============================================================
    # SELECT BEST PLATES PER TRACK
    # ============================================================

    selected_plates = []

    for track_id, candidates in track_candidates.items():

        best_plate = choose_best_plate(
            candidates
        )

        if not best_plate:
            continue

        confidence = track_confidence.get(
            track_id,
            0.0
        )

        selected_plates.append(
            {
                "plate_text": best_plate,
                "track_id": track_id,
                "confidence": confidence,
                "votes": candidates.count(
                    best_plate
                )
            }
        )

        print(
            f"[INFO] Best plate for Track ID="
            f"{track_id}: {best_plate} "
            f"| Votes={candidates.count(best_plate)}"
        )

    # ============================================================
    # GLOBAL PLATE DEDUPLICATION
    # ============================================================

    final_candidates = {}

    for detection in selected_plates:

        plate = detection[
            "plate_text"
        ]

        # If same plate appears with different
        # tracking IDs, keep the strongest result.
        if plate not in final_candidates:

            final_candidates[
                plate
            ] = detection

        else:

            existing = final_candidates[
                plate
            ]

            # Prefer more OCR votes
            if detection["votes"] > existing["votes"]:

                final_candidates[
                    plate
                ] = detection

            # If votes equal, prefer stronger YOLO confidence
            elif (
                detection["votes"]
                == existing["votes"]
                and detection["confidence"]
                > existing["confidence"]
            ):

                final_candidates[
                    plate
                ] = detection

    # ============================================================
    # SAVE FINAL RESULTS TO MYSQL
    # ============================================================

    detections = []

    saved_plates = set()

    for detection in final_candidates.values():

        plate_text = detection[
            "plate_text"
        ]

        track_id = detection[
            "track_id"
        ]

        confidence = detection[
            "confidence"
        ]

        # --------------------------------------------------------
        # Duplicate protection
        # --------------------------------------------------------

        if plate_text in saved_plates:

            print(
                f"[INFO] Duplicate plate ignored: "
                f"{plate_text}"
            )

            continue

        timestamp = datetime.now()

        # --------------------------------------------------------
        # Save to MySQL
        # --------------------------------------------------------

        success = save_detection(
            plate_text=plate_text,
            track_id=int(track_id),
            confidence=float(confidence),
            timestamp=timestamp,
            source=source
        )

        if not success:

            print(
                f"[ERROR] Failed to save plate: "
                f"{plate_text}"
            )

            continue

        saved_plates.add(
            plate_text
        )

        # --------------------------------------------------------
        # API response
        # --------------------------------------------------------

        detection_response = {
            "plate_text": plate_text,
            "track_id": int(track_id),
            "confidence": float(confidence),
            "timestamp": timestamp.isoformat(),
            "source": source
        }

        detections.append(
            detection_response
        )

        print(
            f"[INFO] Saved unique plate: "
            f"Plate={plate_text}, "
            f"Track ID={track_id}, "
            f"Confidence={confidence:.4f}, "
            f"Votes={detection['votes']}"
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print(
        f"[INFO] Final unique plates detected: "
        f"{len(detections)}"
    )

    for detection in detections:

        print(
            f"[INFO] Final result: "
            f"{detection['plate_text']}"
        )

    return detections