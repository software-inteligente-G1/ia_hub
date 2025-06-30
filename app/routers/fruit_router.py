from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from app.services.fruit_service import predict_resnet_image
import io

router = APIRouter()

@router.post("/resnet")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = predict_resnet_image(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
