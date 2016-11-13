'''
The script will create augmented images from the original 8000 training images. It will then 
store all the augmented and original images (training and test images) into a new folder. The 
script will then generate a big dataframe from the new folder by converting each image into 
each row using resizing and graying. 

WARNING: PART 2 might take hours to run! Please set the AUGMENTED_TIMES reasonably. 

Your directory: 
./roof_images - all original 21999 images (please run remove_useless_images.py first!) 
./roof_augmented - this directory will be automated created for you!
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from scipy.misc import imread
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import shutil
import scipy.misc
import pandas as pd
from datetime import datetime

## matlab weights, and other constants
WEIGHT1 = 0.299
WEIGHT2 = 0.587
WEIGHT3 = 0.114
SCALE_AFTER_RESIZE = 64
AUGMENTED_TIMES = 5

train_ids = pd.read_csv('id_train.csv')
train_ids_set = set(train_ids['Id'].values)

## paths
path_of_input = './roof_images'  ## dir of input raw images - make sure it has only 21,999 images!
path_of_output = './roof_augmented'
if os.path.exists(path_of_output): ## remove dir if exists
    shutil.rmtree(path_of_output)
os.makedirs(path_of_output)
allOriginTrainFiles = [path_of_input + '/' + f for f in listdir(path_of_input) if int(f.split('.')[0]) in train_ids_set and isfile(join(path_of_input, f))]

def weightedAverage(pixel):
    return (WEIGHT1*pixel[0] + WEIGHT2*pixel[1] + WEIGHT3*pixel[2]) / 255.0
    
datagen = ImageDataGenerator(
        rotation_range=7,
        width_shift_range=0.1,
        height_shift_range=0.1,
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        
print("PART 2 - Started augmentation and saving all images to path: ", path_of_output)
for i, file in enumerate(allOriginTrainFiles):
    if i % 100 == 0:
        print("Processed file ", i)
    prefix = file.split('/')[-1].split('.')[0]
    img = load_img(file)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    j = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir=path_of_output, save_prefix=prefix, save_format='jpeg'):
        j += 1
        if j >= AUGMENTED_TIMES:
            break  
 ## copy paste all files (original images) in roof_images folder to the new folder 
allOriginFiles = [path_of_input + '/' + f for f in listdir(path_of_input) if isfile(join(path_of_input, f))]
for file in allOriginFiles:
    shutil.copyfile(file, path_of_output + '/' + file.split('/')[-1].split('.')[0] + '.jpg')
      