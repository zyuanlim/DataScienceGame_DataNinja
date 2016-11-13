import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join

train_id = pd.read_csv('../Competition_Data/id_train.csv')
test_id = pd.read_csv('../Competition_Data/sample_submission4.csv')
path_of_folder = '../roof_images'

total_ids = set(train_id.Id.values).union(set(test_id.Id.values))

allImageFiles = [path_of_folder + '/' + f for f in listdir(path_of_folder) if isfile(join(path_of_folder, f))]

for i, image in enumerate(allImageFiles):
    if i % 100 == 0:
        print(i)
    temp = image.split('/')[-1].split('.')[0]
    if int(temp) not in total_ids:
        print("Removing ", image)
        os.remove(image)