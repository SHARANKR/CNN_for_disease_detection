from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from app.data_preprocess.dividing_dataset_to_x_and_y_train import x_train, y_train

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(
    enumerate(weights)
)

print(class_weights)