import os
from app.data_import.data_import_file import train_data

length = os.listdir(train_data)
print(length)