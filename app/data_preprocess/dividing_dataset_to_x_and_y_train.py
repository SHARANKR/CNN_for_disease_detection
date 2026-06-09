from keras.utils import image_dataset_from_directory
from app.data_import.data_import_file import train_data
import numpy as np

train_ds = image_dataset_from_directory(
    train_data,
    image_size=(224,224),
    batch_size=32,
    shuffle=True
)

x_train = []
y_train = []

for images, labels in train_ds:

    x_train.append(images.numpy())
    y_train.append(labels.numpy())

x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

print(x_train.shape)
print(y_train.shape)