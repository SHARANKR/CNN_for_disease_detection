from app.data_import.data_import_file import train_data

import os
import matplotlib.pyplot as plt

train_path = train_data

classes = os.listdir(train_path)

counts = []

for cls in classes:
    counts.append(
        len(os.listdir(os.path.join(train_path, cls)))
    )

plt.bar(classes, counts)

plt.xlabel("Classes")
plt.ylabel("Images")

plt.show()