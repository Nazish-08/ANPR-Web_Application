from fastapi import APIRouter, HTTPException

from app.database import get_detections, get_detection_by_id


router = APIRouter(
    prefix="/detections",
    tags=["Detections"]
)


@router.get("/")
def list_detections():
    """
    Return all stored detections.
    Newest detections are returned first.
    """

    detections = get_detections()

    return {
        "total": len(detections),
        "detections": detections
    }


@router.get("/{detection_id}")
def get_single_detection(detection_id: int):
    """
    Return one detection by database ID.
    """

    detection = get_detection_by_id(detection_id)

    if detection is None:
        raise HTTPException(
            status_code=404,
            detail="Detection not found"
        )

    return detection