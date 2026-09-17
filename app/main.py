from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers import detect
from app.routers.detections import router as detections_router


app = FastAPI(
    title="Number Plate Detection API",
    description="YOLOv12 + PaddleOCR based ANPR service",
    version="1.0.0"
)


# ============================================
# API ROUTERS
# ============================================

app.include_router(
    detect.router,
    prefix="/detect",
    tags=["Detection"]
)

app.include_router(
    detections_router
)


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "anpr-api"
    }


# ============================================
# FRONTEND
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# Serve frontend files
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


# Serve cropped plate images
CROPS_DIR = BASE_DIR / "data" / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/crops",
    StaticFiles(directory=CROPS_DIR),
    name="crops"
)


# Home page
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )