import os
import yaml

with open('app/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
train_data = config['data_path']['train_data']
test_data = config['data_path']['test_data']

# print(train_data)