from fastapi import FastAPI

from app.routers import detect
from app.routers.detections import router as detections_router


app = FastAPI(
    title="Number Plate Detection API",
    description="YOLOv12 + PaddleOCR based ANPR service",
    version="1.0.0"
)


# Register routers
app.include_router(
    detect.router,
    prefix="/detect",
    tags=["Detection"]
)

app.include_router(
    detections_router
)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "anpr-api"
    }