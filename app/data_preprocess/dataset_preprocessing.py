#balancing the data imbalance after data analytics
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from app.data_import.data_import_file import train_data

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_path = train_data

source_dir = os.path.join(train_path, "COVID19")
save_dir = os.path.join(train_path, "COVID19")

for img_name in os.listdir(source_dir):

    img = load_img(os.path.join(source_dir, img_name))
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)

    i = 0

    for batch in datagen.flow(
        img,
        batch_size=1,
        save_to_dir=save_dir,
        save_prefix="aug",
        save_format="jpg"
    ):
        i += 1

        if i >= 5:
            break