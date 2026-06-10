import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from app.data_preprocess.dividing_dataset_to_x_and_y_train import x_train, y_train
from app.data_preprocess.class_weights_to_balance_classes import class_weights


save_dir = r'C:\Users\win10\Desktop\New folder (6)\CNN_for_disease_detection\app\cnn_models'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'my_model_1.keras')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu',))
model.add(MaxPool2D(pool_size=(2,2), padding='valid', strides=1))
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='valid', strides=1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    class_weight=class_weights,
    epochs=5,
    batch_size=32
)
model.save(save_path)
