from keras.utils import image_dataset_from_directory
from app.data_import.data_import_file import test_data
import numpy as np

train_ds = image_dataset_from_directory(
    test_data,
    image_size=(224,224),
    batch_size=32,
    shuffle=True
)

x_test = []
y_test = []

for images, labels in train_ds:

    x_test.append(images.numpy())
    y_test.append(labels.numpy())

x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)

print(x_test.shape)
print(y_test.shape)