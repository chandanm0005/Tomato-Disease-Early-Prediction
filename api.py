import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Tomato Disease Detector")


@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    with open(BASE_DIR / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image upload and return disease prediction results.
    Requires a trained model at models/best_model.pth.
    """
    # Lazy import so the app starts even without torch installed
    try:
        from infer_gradcam import infer
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference dependencies not installed: {e}"
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    model_path = BASE_DIR / "models" / "best_model.pth"
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run train_tomato.py first to train and save the model."
        )

    # Save upload to a temp file then run inference
    import tempfile, os
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = infer(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=result)
