import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pathlib import Path
import re


# ============================================================
# PATHS & MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

print("[INFO] Loading YOLO model...")
model = YOLO(str(MODEL_PATH))
names = model.names

print("[INFO] Loading PaddleOCR...")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

print("[INFO] Models loaded successfully.")


# ============================================================
# BASIC GEOMETRY
# ============================================================

def iou(box_a, box_b):
    """Calculate Intersection over Union."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def box_size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def vertical_overlap_ratio(box_a, box_b):
    """Vertical overlap divided by the smaller box height."""
    _, ay1, _, ay2 = box_a
    _, by1, _, by2 = box_b

    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    ha = max(1, ay2 - ay1)
    hb = max(1, by2 - by1)

    return overlap / min(ha, hb)


def horizontal_gap(box_a, box_b):
    """Gap between boxes on the X axis. 0 when they overlap."""
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    if ax2 < bx1:
        return bx1 - ax2

    if bx2 < ax1:
        return ax1 - bx2

    return 0


def center_y(box):
    return (box[1] + box[3]) / 2.0


# ============================================================
# MERGE FRAGMENTED PLATE DETECTIONS
# ============================================================

def should_merge_plate_boxes(box_a, box_b):
    """
    Decide whether two YOLO detections are probably pieces
    of the same number plate.

    This fixes the exact problem seen in the supplied images:
    the model was detecting the left, middle and right portions
    of one plate as separate boxes.
    """

    wa, ha = box_size(box_a)
    wb, hb = box_size(box_b)

    min_h = min(ha, hb)

    # Strong vertical alignment is expected for plate fragments.
    v_overlap = vertical_overlap_ratio(box_a, box_b)
    y_difference = abs(center_y(box_a) - center_y(box_b))

    if v_overlap >= 0.45:
        aligned = True
    else:
        aligned = y_difference <= 0.55 * min_h

    if not aligned:
        return False

    # Allow a small horizontal gap between fragments.
    gap = horizontal_gap(box_a, box_b)

    if gap > 0.65 * min_h:
        return False

    # Do not merge boxes that are wildly different in height.
    height_ratio = min(ha, hb) / max(ha, hb)

    if height_ratio < 0.45:
        return False

    return True


def merge_two_boxes(box_a, box_b):
    """Return the union of two bounding boxes."""
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def merge_fragmented_boxes(detections):
    """
    Merge multiple overlapping/adjacent detections belonging
    to the same physical number plate.
    """

    if not detections:
        return []

    # Highest confidence first.
    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    clusters = []

    for detection in detections:
        current_box = detection["bbox"]
        current_conf = detection["confidence"]
        merged_into_existing = False

        for cluster in clusters:
            if should_merge_plate_boxes(
                current_box,
                cluster["bbox"]
            ):
                cluster["bbox"] = merge_two_boxes(
                    cluster["bbox"],
                    current_box
                )

                cluster["confidence"] = max(
                    cluster["confidence"],
                    current_conf
                )

                cluster["parts"] += 1
                merged_into_existing = True
                break

        if not merged_into_existing:
            clusters.append({
                "bbox": current_box[:],
                "confidence": current_conf,
                "class": detection["class"],
                "parts": 1,
            })

    # A second pass handles chains such as:
    # left -> middle -> right.
    changed = True

    while changed:
        changed = False
        result = []

        for cluster in clusters:
            merged = False

            for existing in result:
                if should_merge_plate_boxes(
                    cluster["bbox"],
                    existing["bbox"]
                ):
                    existing["bbox"] = merge_two_boxes(
                        existing["bbox"],
                        cluster["bbox"]
                    )

                    existing["confidence"] = max(
                        existing["confidence"],
                        cluster["confidence"]
                    )

                    existing["parts"] += cluster["parts"]
                    merged = True
                    changed = True
                    break

            if not merged:
                result.append(cluster)

        clusters = result

    # Remove internal bookkeeping.
    for cluster in clusters:
        cluster.pop("parts", None)

    return clusters


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_plates(frame):
    """
    Detect number plates.

    The model can return several fragments for one plate.
    Those fragments are merged before OCR.
    """

    results = model.predict(
        frame,
        verbose=False,

        # 0.35 keeps difficult/angled plates while rejecting
        # very weak detections.
        conf=0.35,

        # YOLO NMS.
        iou=0.45,

        max_det=50
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    detections = []

    for box, conf, class_id in zip(
        boxes,
        confs,
        class_ids
    ):
        label = names[class_id]

        if label.lower() != "numberplate":
            continue

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            "confidence": float(conf),
            "class": label
        })

    # IMPORTANT:
    # Do not use the old "larger box wins" logic here.
    # That logic was discarding useful fragments before they
    # could be combined into the complete plate.
    return merge_fragmented_boxes(detections)


# ============================================================
# CROP WITH PADDING
# ============================================================

def crop_plate(frame, bbox, padding=0.05):
    """
    Crop plate with small padding.
    """

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return frame[y1:y2, x1:x2]


# ============================================================
# ORDER FOUR CORNER POINTS
# ============================================================

def order_points(points):
    """
    Order four points as:
    top-left, top-right, bottom-right, bottom-left.
    """

    pts = np.asarray(points, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ],
        dtype=np.float32
    )


# ============================================================
# PERSPECTIVE RECTIFICATION
# ============================================================

def perspective_transform(image, points):
    """
    Straighten a detected quadrilateral plate.
    """

    rect = order_points(points)

    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)

    width = int(max(width_top, width_bottom))
    height = int(max(height_right, height_left))

    if width < 100 or height < 20:
        return None

    # Indian plates are normally wider than tall.
    # If the detected rectangle is vertical, rotate it.
    if width < height:
        width, height = height, width

    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC
    )

    return warped


def find_plate_quadrilateral(image):
    """
    Try to find the physical rectangular plate inside the YOLO crop.

    This is useful when the plate is tilted or viewed in perspective.
    """

    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        60,
        180
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 3)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]
    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.10:
            continue

        if area > image_area * 0.98:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        points = approx.reshape(4, 2).astype(
            np.float32
        )

        rect = cv2.minAreaRect(points)
        rw, rh = rect[1]

        if rw <= 0 or rh <= 0:
            continue

        ratio = max(rw, rh) / min(rw, rh)

        # Number plates are normally clearly wider than tall.
        if ratio < 1.7 or ratio > 7.5:
            continue

        score = area * min(ratio, 5.0)

        candidates.append(
            (score, points)
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0][1]


# ============================================================
# ROTATION FALLBACK
# ============================================================

def rotate_to_horizontal(image):
    """
    Estimate the plate angle and rotate it horizontally.
    """

    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(20, image.shape[1] // 8),
        minLineLength=max(30, image.shape[1] // 5),
        maxLineGap=20
    )

    if lines is None:
        return image

    angles = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = np.degrees(
            np.arctan2(dy, dx)
        )

        # Keep mostly horizontal lines.
        if -25 <= angle <= 25:
            angles.append(angle)

    if not angles:
        return image

    angle = float(np.median(angles))

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_variants(image):
    """
    Generate several OCR-friendly versions.

    OCR is tested against multiple versions because a single
    preprocessing method is not reliable for every plate.
    """

    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]

    # Make the plate large enough for OCR.
    target_width = 900

    if w < target_width:
        scale = target_width / w

        image = cv2.resize(
            image,
            (
                int(w * scale),
                int(h * scale)
            ),
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    denoised = cv2.bilateralFilter(
        gray,
        7,
        50,
        50
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        image,
        gray,
        enhanced,
        sharpened,
        binary
    ]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_plate_text(text):
    """
    Remove spaces and OCR noise.
    """

    if not text:
        return ""

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    # HSRP/Indian plates often produce IND as a separate OCR
    # token. Remove it when it appears as a standalone prefix.
    if text.startswith("IND") and len(text) > 5:
        text = text[3:]

    return text


def plate_text_score(text, confidence):
    """
    Rank OCR candidates.

    Longer alphanumeric plate strings are generally more useful
    than tiny fragments such as IND or 265.
    """

    if not text:
        return -1.0

    length_bonus = min(len(text), 10) * 0.04

    # Penalize extremely short OCR fragments.
    short_penalty = 0.25 if len(text) < 4 else 0.0

    return (
        confidence
        + length_bonus
        - short_penalty
    )


# ============================================================
# OCR
# ============================================================

def run_ocr(image):
    """
    Run PaddleOCR once and return:
    (cleaned_text, confidence)
    """

    try:
        result = ocr.predict(image)

        if not result:
            return "", 0.0

        data = result[0]

        if "rec_texts" not in data:
            return "", 0.0

        texts = data["rec_texts"]
        scores = data.get(
            "rec_scores",
            []
        )

        if not texts:
            return "", 0.0

        raw_text = " ".join(
            str(text)
            for text in texts
        )

        cleaned = clean_plate_text(
            raw_text
        )

        if not cleaned:
            return "", 0.0

        if scores:
            confidence = float(
                np.mean(scores)
            )
        else:
            confidence = 0.0

        return cleaned, confidence

    except Exception as error:
        print(
            f"[WARNING] OCR failed: {error}"
        )
        return "", 0.0


def read_plate(cropped_image):
    """
    Robust OCR pipeline.

    Tests:
    1. Original crop
    2. Rotated crop
    3. Perspective-corrected crop
    4. Multiple preprocessing variants

    The best useful result is returned.
    """

    if cropped_image is None:
        return "", 0.0

    if cropped_image.size == 0:
        return "", 0.0

    candidates = []

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    candidates.append(
        cropped_image.copy()
    )

    # --------------------------------------------------------
    # ROTATION CORRECTION
    # --------------------------------------------------------

    rotated = rotate_to_horizontal(
        cropped_image
    )

    if rotated is not None:
        candidates.append(rotated)

    # --------------------------------------------------------
    # PERSPECTIVE CORRECTION
    # --------------------------------------------------------

    quad = find_plate_quadrilateral(
        cropped_image
    )

    if quad is not None:

        warped = perspective_transform(
            cropped_image,
            quad
        )

        if warped is not None:
            candidates.append(warped)

    # Also try perspective correction after rotation.
    if rotated is not None:

        rotated_quad = find_plate_quadrilateral(
            rotated
        )

        if rotated_quad is not None:

            warped_rotated = perspective_transform(
                rotated,
                rotated_quad
            )

            if warped_rotated is not None:
                candidates.append(
                    warped_rotated
                )

    best_text = ""
    best_confidence = 0.0
    best_score = -1.0

    # --------------------------------------------------------
    # OCR ALL CANDIDATES
    # --------------------------------------------------------

    for candidate in candidates:

        for variant in preprocess_variants(
            candidate
        ):

            text, confidence = run_ocr(
                variant
            )

            score = plate_text_score(
                text,
                confidence
            )

            if score > best_score:

                best_score = score
                best_text = text
                best_confidence = confidence

    return best_text, best_confidence


# ============================================================
# COMPLETE PLATE PROCESSING
# ============================================================

def process_plate(frame, detection):
    """
    YOLO bbox
        ↓
    Fragment-merged bbox
        ↓
    Padded crop
        ↓
    Rotation/perspective correction
        ↓
    OCR preprocessing
        ↓
    PaddleOCR
        ↓
    Clean plate number
    """

    bbox = detection["bbox"]

    crop = crop_plate(
        frame,
        bbox,
        padding=0.05
    )

    text, ocr_confidence = read_plate(
        crop
    )

    return {
        "text": text,
        "bbox": bbox,
        "detection_confidence": detection[
            "confidence"
        ],
        "ocr_confidence": ocr_confidence
    }


# ============================================================
# COMPLETE IMAGE PROCESSING
# ============================================================

def process_image(frame):
    """
    Detect and read all number plates in an image.
    """

    detections = detect_plates(
        frame
    )

    plates = []

    for detection in detections:

        result = process_plate(
            frame,
            detection
        )

        plates.append(result)

    return plates
