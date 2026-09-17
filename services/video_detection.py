import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pathlib import Path
import re
from datetime import datetime
from collections import Counter, defaultdict

from app.database import save_detection


# ============================================================
# PATHS & MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

print("[INFO] Loading YOLO model (video)...")
model = YOLO(str(MODEL_PATH))
names = model.names

print("[INFO] Loading PaddleOCR (video)...")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

print("[INFO] Video models loaded successfully.")


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def box_size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def vertical_overlap_ratio(box_a, box_b):
    _, ay1, _, ay2 = box_a
    _, by1, _, by2 = box_b
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    ha = max(1, ay2 - ay1)
    hb = max(1, by2 - by1)
    return overlap / min(ha, hb)


def horizontal_gap(box_a, box_b):
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b
    if ax2 < bx1:
        return bx1 - ax2
    if bx2 < ax1:
        return ax1 - bx2
    return 0


def center_y(box):
    return (box[1] + box[3]) / 2.0


def should_merge_plate_boxes(box_a, box_b):
    wa, ha = box_size(box_a)
    wb, hb = box_size(box_b)
    min_h = min(ha, hb)

    v_overlap = vertical_overlap_ratio(box_a, box_b)
    y_difference = abs(center_y(box_a) - center_y(box_b))

    if v_overlap >= 0.45:
        aligned = True
    else:
        aligned = y_difference <= 0.55 * min_h

    if not aligned:
        return False

    gap = horizontal_gap(box_a, box_b)
    if gap > 0.65 * min_h:
        return False

    height_ratio = min(ha, hb) / max(ha, hb)
    if height_ratio < 0.45:
        return False

    return True


def merge_two_boxes(box_a, box_b):
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def merge_fragmented_boxes(detections):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    clusters = []

    for detection in detections:
        current_box = detection["bbox"]
        current_conf = detection["confidence"]
        merged = False

        for cluster in clusters:
            if should_merge_plate_boxes(current_box, cluster["bbox"]):
                cluster["bbox"] = merge_two_boxes(cluster["bbox"], current_box)
                cluster["confidence"] = max(cluster["confidence"], current_conf)
                cluster["parts"] += 1
                merged = True
                break

        if not merged:
            clusters.append({
                "bbox": current_box[:],
                "confidence": current_conf,
                "class": detection["class"],
                "parts": 1,
            })

    for cluster in clusters:
        cluster.pop("parts", None)

    return clusters


# ============================================================
# CROP
# ============================================================

def crop_plate(frame, bbox, padding=0.05):
    """Safely crop plate, clip to frame bounds."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Clip coordinates to frame
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    # Sanity check
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop

# ============================================================
# tight_crop_plate 
# ============================================================

def tight_crop_plate(plate_crop):
    """
    Tightly crop to just the number plate.
    Detects bright white plate region and crops to it.
    Removes black patches and background.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    h, w = plate_crop.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # ✅ Remove black patches first — set very dark pixels to medium gray
    # (so they don't interfere with detection)
    gray = np.where(gray < 30, 128, gray).astype(np.uint8)

    # ✅ Threshold to find BRIGHT region (white plate)
    # White plate = bright pixels (>180), black patch = dark (<30)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # Morphological close to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return plate_crop

    # Find best rectangular contour (largest, plate-like aspect ratio)
    best_rect = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Minimum area check — at least 15% of crop
        if area < (w * h) * 0.15:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        
        if ch == 0:
            continue
        
        aspect = cw / float(ch)

        # Number plate aspect ratio: 2:1 to 6:1
        if aspect < 1.8 or aspect > 6.5:
            continue

        # Score = area * aspect — bigger + more plate-like wins
        score = area * min(aspect, 5.0)

        if score > best_score:
            best_score = score
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        return plate_crop

    x, y, cw, ch = best_rect

    # Small padding
    pad = 3
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)

    tight = plate_crop[y1:y2, x1:x2]

    if tight is None or tight.size == 0:
        return plate_crop

    return tight

# ============================================================
# PREPROCESSING — all 3-channel BGR for PaddleOCR
# ============================================================

def preprocess_variants(image):
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    target_width = 900

    if w < target_width:
        scale = target_width / w
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    # Convert all grayscale/binary back to 3-channel BGR
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    sharpened_3ch = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return [image, gray_3ch, enhanced_3ch, sharpened_3ch, binary_3ch]


# ============================================================
# TEXT CLEANING & VALIDATION
# ============================================================

def clean_plate_text(text):
    if not text:
        return ""
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    if text.startswith("IND") and len(text) > 5:
        text = text[3:]
    return text


def is_plausible_indian_plate(text):
    """
    Strict Indian plate format validation.
    Filters garbage like SAIDCS, EER, 1 etc.
    """
    if not text:
        return False
    
    text = clean_plate_text(text)
    
    if len(text) < 8 or len(text) > 12:
        return False
    
    # Normal Indian format: MH12AB1234
    normal_pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"
    if re.fullmatch(normal_pattern, text):
        return True
    
    # Bharat series: BH12AB1234
    bharat_pattern = r"^BH[0-9]{2}[A-Z]{1,2}[0-9]{4}$"
    if re.fullmatch(bharat_pattern, text):
        return True
    
    return False


def plate_text_score(text, confidence):
    if not text:
        return -1.0
    if not is_plausible_indian_plate(text):
        return -1.0
    return float(confidence) + min(len(text), 10) * 0.05


# ============================================================
# OCR
# ============================================================

def run_ocr(image):
    if image is None or image.size == 0:
        return "", 0.0

    try:
        result = ocr.predict(image)

        if not result:
            return "", 0.0

        data = result[0]

        if isinstance(data, dict):
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
        else:
            texts = getattr(data, "rec_texts", []) or []
            scores = getattr(data, "rec_scores", []) or []

        if not texts:
            return "", 0.0

        raw_text = " ".join(str(t) for t in texts)
        cleaned = clean_plate_text(raw_text)

        if not cleaned:
            return "", 0.0

        confidence = float(np.mean(scores)) if scores else 0.0
        return cleaned, confidence

    except Exception as error:
        print(f"[WARNING] OCR failed: {error}")
        return "", 0.0


def read_plate(cropped_image):
    """Simple OCR — one call only, like original."""
    if cropped_image is None or cropped_image.size == 0:
        return "", 0.0

    try:
        result = ocr.predict(cropped_image)
        data = result[0] if result else None
        
        if not data or "rec_texts" not in data:
            return "", 0.0
        
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        
        if not texts:
            return "", 0.0
        
        raw_text = " ".join(str(t) for t in texts)
        cleaned = clean_plate_text(raw_text)
        
        if not cleaned:
            return "", 0.0
        
        confidence = float(np.mean(scores)) if scores else 0.0
        return cleaned, confidence
    
    except Exception as error:
        print(f"[WARNING] OCR failed: {error}")
        return "", 0.0


# ============================================================
# VIDEO PROCESSING — this is the actual process_video()
# ============================================================

VIDEO_SAMPLE_SECONDS = 1.0
VIDEO_FRAME_WIDTH = 800
VIDEO_FRAME_HEIGHT = 480
VIDEO_MIN_OCR_CONFIDENCE = 0.5


def _normalise_plate_for_comparison(text):
    return clean_plate_text(text).upper()


def _plate_similarity(a, b):
    a = _normalise_plate_for_comparison(a)
    b = _normalise_plate_for_comparison(b)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 4 or len(b) < 4:
        return 0.0

    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix += 1

    suffix = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        suffix += 1

    return max(
        prefix / max(len(a), len(b)),
        suffix / max(len(a), len(b)),
    )


def _choose_best_plate(votes):
    valid_votes = [v for v in votes if is_plausible_indian_plate(v["text"])]

    if not valid_votes:
        return None

    grouped = defaultdict(list)

    for item in valid_votes:
        text = item["text"]
        matched_key = None

        for key in grouped:
            if _plate_similarity(text, key) >= 0.82:
                matched_key = key
                break

        if matched_key is None:
            grouped[text].append(item)
        else:
            grouped[matched_key].append(item)

    best_group = None
    best_rank = None

    for key, items in grouped.items():
        counts = Counter(item["text"] for item in items)
        most_common_text, exact_votes = counts.most_common(1)[0]

        avg_conf = sum(item["ocr_confidence"] for item in items) / len(items)
        max_conf = max(item["ocr_confidence"] for item in items)

        rank = (exact_votes, len(items), avg_conf, max_conf, len(most_common_text))

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_group = items

    if not best_group:
        return None

    exact_counts = Counter(item["text"] for item in best_group)
    final_text = exact_counts.most_common(1)[0][0]

    final_item = max(
        [item for item in best_group if item["text"] == final_text],
        key=lambda item: item["ocr_confidence"]
    )

    return {
        "text": final_text,
        "bbox": final_item["bbox"],
        "detection_confidence": final_item["detection_confidence"],
        "ocr_confidence": final_item["ocr_confidence"],
        "track_id": final_item["track_id"],
        "frame_number": final_item["frame_number"],
        "crop": final_item["crop"],
        "votes": exact_counts[final_text],
    }


def process_video(video_path: str, source: str):
    """Detect plates across a video, sample ~5 frames per second."""
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / fps) if total_frames > 0 else 0.0

    frame_interval = max(1, int(round(fps * VIDEO_SAMPLE_SECONDS)))

    print("=" * 60)
    print(f"[VIDEO] Processing: {source}")
    print(f"[VIDEO] FPS: {fps:.2f}")
    print(f"[VIDEO] Total frames: {total_frames}")
    print(f"[VIDEO] Duration: {duration:.2f} sec")
    print(f"[VIDEO] Sampling every {frame_interval} frames (~5 FPS)")
    print("=" * 60)

    track_votes = defaultdict(list)

    frame_count = 0
    sampled_frames = 0

    crop_dir = BASE_DIR / "data" / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[VIDEO] Video ended at frame {frame_count}")
                break

            frame_count += 1

            if frame_count % frame_interval != 0:
                continue

            sampled_frames += 1

            if sampled_frames == 1 or sampled_frames % 5 == 0:
                print(f"[VIDEO] Sampled frame #{sampled_frames} (orig {frame_count})")

            frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.35,
                iou=0.45,
                max_det=50,
            )

            if not results:
                continue

            result = results[0]

            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue

            ids = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().numpy()

            for track_id, box, class_id, confidence in zip(ids, boxes, class_ids, confidences):
                label = names[class_id]

                if label.lower() != "numberplate":
                    continue

                confidence = float(confidence)

                if confidence < 0.50:
                    continue

                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = [x1, y1, x2, y2]
                crop = crop_plate(frame, bbox, padding=0.0)   # ← padding 0

                if crop is None or crop.size == 0:
                    continue



                text, ocr_confidence = read_plate(crop)

                if not is_plausible_indian_plate(text):
                    continue

                if ocr_confidence < VIDEO_MIN_OCR_CONFIDENCE:
                    continue

                print(f"[VIDEO] Valid plate: {text} | track={track_id} | conf={ocr_confidence:.3f}")

                observation = {
                    "text": text,
                    "bbox": bbox,
                    "detection_confidence": confidence,
                    "ocr_confidence": float(ocr_confidence),
                    "track_id": int(track_id),
                    "frame_number": frame_count,
                    "crop": crop.copy(),
                }

                track_votes[int(track_id)].append(observation)

    finally:
        cap.release()

    print(f"[VIDEO] Sampled frames processed: {sampled_frames}")

    best_by_track = {}
    for track_id, votes in track_votes.items():
        best = _choose_best_plate(votes)
        if best is not None:
            best_by_track[track_id] = best

    # Deduplicate across tracks
    final_by_plate = {}
    for track_id, item in best_by_track.items():
        plate = item["text"]
        existing = final_by_plate.get(plate)
        if existing is None:
            final_by_plate[plate] = item
            continue

        existing_rank = (existing["votes"], existing["ocr_confidence"], existing["detection_confidence"])
        current_rank = (item["votes"], item["ocr_confidence"], item["detection_confidence"])

        if current_rank > existing_rank:
            final_by_plate[plate] = item

    detections = []

    for plate_text, item in final_by_plate.items():
        crop_filename = (
            f"{re.sub(r'[^A-Z0-9]', '', plate_text)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )

        crop_path = crop_dir / crop_filename

        if not cv2.imwrite(str(crop_path), item["crop"]):
            print(f"[ERROR] Failed to save crop: {crop_path}")
            continue

        timestamp = datetime.now()

        save_detection(
            plate_text=plate_text,
            track_id=int(item["track_id"]),
            confidence=float(item["ocr_confidence"]),
            timestamp=timestamp,
            source=source,
        )

        crop_url = f"/crops/{crop_filename}"

        detection = {
            "plate_text": plate_text,
            "track_id": int(item["track_id"]),
            "confidence": float(item["ocr_confidence"]),
            "detection_confidence": float(item["detection_confidence"]),
            "timestamp": timestamp.isoformat(),
            "source": source,
            "votes": int(item["votes"]),
            "crop_image": crop_url,
        }

        detections.append(detection)
        print(f"[VIDEO] Saved plate: {plate_text} | votes={item['votes']}")

    print(f"[VIDEO] Final unique plates: {len(detections)}")

    return detections