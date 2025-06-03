from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from ultralytics import YOLO
import numpy as np
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://192.168.1.19:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv11 model
try:
    model = YOLO("yolo11n (1).pt")  # Ensure this path is correct
    logger.info("YOLOv11 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise Exception(f"Model loading failed: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Log file details
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}, size: {file.size} bytes")

        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            error_msg = f"Invalid file type: {file.content_type}. Only JPEG and PNG are supported."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Read and validate image file
        image_bytes = await file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        if len(image_bytes) == 0:
            error_msg = "Empty file received"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            error_msg = f"Cannot identify image file: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Perform prediction
        results = model.predict(image, conf=0.5)

        # Process results
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            detection = {
                "class": results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            }
            detections.append(detection)
        logger.info(f"Detected {len(detections)} objects")

        # Convert image to base64 for visualization
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "detections": detections,
            "image": f"data:image/jpeg;base64,{img_str}"
        }
    except HTTPException as he:
        logger.error(f"HTTP error: {str(he.detail)}")
        raise he
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")