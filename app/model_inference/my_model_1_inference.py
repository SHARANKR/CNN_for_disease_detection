from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np


# Load model
model = load_model(
    r'C:\Users\win10\Desktop\New folder (6)\CNN_for_disease_detection\app\cnn_models\my_model_1.keras'
)

# Image path
img_path = (
    r'C:\Users\win10\Desktop\New folder (6)\CNN_for_disease_detection'
    r'\app\dataset\test\COVID19\COVID19(460).jpg'
)

# Load image
img = load_img(
    img_path,
    target_size=(224, 224)
)

# Convert to numpy array
img = img_to_array(img)

# Normalize (same as training)
img = img.astype(np.float32) / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

print("Input Shape:", img.shape)

# Predict
result = model.predict(img)

print("Raw Prediction:", result)

# Get class index
predicted_class = np.argmax(result, axis=1)[0]

# Class mapping
classes = {
    0: "COVID19",
    1: "NORMAL",
    2: "PNEUMONIA"
}

print("Predicted Class:", classes[predicted_class])

# Confidence
confidence = np.max(result)

print("Confidence:", confidence)