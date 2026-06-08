import numpy as np
from app.data_import.data_import_file import train_data
import os
import cv2

dataset_path = train_data

sum_pixels = 0
sum_squared_pixels = 0
num_pixels = 0

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img = img.astype(np.float32) / 255.0

        sum_pixels += np.sum(img)

        sum_squared_pixels += np.sum(img ** 2)

        num_pixels += img.size

mean = sum_pixels / num_pixels

std = np.sqrt(
    (sum_squared_pixels / num_pixels)
    - (mean ** 2)
)

print("Mean:", mean)
print("Std :", std)