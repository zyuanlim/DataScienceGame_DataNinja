```
This script will move the augmented and original images (only training set) into a new folder within the roof_augmented 
folder called organised, then split them into 75% train folder and 25% val folder (according to fb resnet naming convention)
```

from os import listdir
from os.path import isfile, join
import os
import shutil


path_of_original_folder = './roof_augmented'
path_of_organised_folder = './roof_augmented/organised'
if os.path.exists(path_of_organised_folder): ## remove dir if exists
    shutil.rmtree(path_of_organised_folder)
os.makedirs(path_of_organised_folder)

for set_folder in ['train', 'val']:
    for label_folder in ['label1', 'label2', 'label3', 'label4']:
        os.makedirs(join(path_of_organised_folder, set_folder, label_folder))

allFiles = [f for f in listdir(path_of_original_folder)]
val_images = train_ids['Id'].sample(frac=0.25, random_state=0)

i = 0
for file in allFiles:
    if file.find('_') > 0:
        file_id = file.split('_')[0]
    elif file.find('.') > 0:
        file_id = file.split('.')[0]
    else:
        continue
    if int(file_id) not in train_ids_set:
        continue
    if i % 100 == 0:
        print("Processed file ", i)
    label = train_ids.ix[train_ids['Id'] == long(file_id), 'label'].values[0]
    if int(file_id) in val_images.values:
        shutil.move(join(path_of_original_folder, file), join(path_of_organised_folder, 'val/label%s/' %label))
    else:
        shutil.move(join(path_of_original_folder, file), join(path_of_organised_folder, 'train/label%s/' %label))
    
    i += 1