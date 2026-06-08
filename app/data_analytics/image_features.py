import cv2
import matplotlib.pyplot as plt
from app.data_import.data_import_file import train_data
import os

img_sample_1 = 'COVID19\COVID19(0).jpg'

img = cv2.imread(os.path.join(train_data, img_sample_1), 0)

plt.hist(img.ravel(), bins=256)

plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.show()