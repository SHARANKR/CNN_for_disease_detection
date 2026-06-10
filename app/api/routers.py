from fastapi import APIRouter, UploadFile, File
import io
from PIL import Image
from keras.utils import img_to_array
import numpy as np
from keras.models import load_model
from app.utils.classes import classes

model = load_model(r'C:\Users\win10\Desktop\New folder (6)\CNN_for_disease_detection\app\cnn_models\my_model_1.keras')

api_router = APIRouter()

@api_router.post('/post_images', tags=['post_images'])
async def upload_image(upload: UploadFile = File(...)):
    img_bytes = await upload.read()
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert('RGB')
    image = image.resize((224,224))
    image = img_to_array(image)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    pred_prob = model.predict(image)
    pred_class = np.argmax(pred_prob, axis=1)[0]
    confidence = float(np.max(pred_prob))
    
    return {
        'File_name': upload.filename,
        'Predicted_class': classes[pred_class],
        'Confidence': confidence,
    }
    