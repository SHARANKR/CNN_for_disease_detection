from keras.models import load_model
from app.data_preprocess.dividing_dataset_to_x_and_y_test import x_test, y_test

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import numpy as np


model = load_model(
    r'C:\Users\win10\Desktop\New folder (6)\CNN_for_disease_detection\app\cnn_models\my_model_1.keras'
)

# Prediction probabilities
y_pred = model.predict(x_test)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred, axis=1)

print("Precision:",
      precision_score(
          y_test,
          y_pred,
          average='weighted'
      ))

print("Recall:",
      recall_score(
          y_test,
          y_pred,
          average='weighted'
      ))

print("F1:",
      f1_score(
          y_test,
          y_pred,
          average='weighted'
      ))

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "COVID19",
            "NORMAL",
            "PNEUMONIA"
        ]
    )
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)