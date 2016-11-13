from scipy.misc import imread
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

WEIGHT1 = 0.299
WEIGHT2 = 0.587
WEIGHT3 = 0.114
SCALE_AFTER_RESIZE = 64

#WEIGHT1 = 0.3333333
#WEIGHT2 = 0.3333333
#WEIGHT3 = 0.3333333

path_of_input = './roof_images'  ## dir of input raw images - make sure it has only 21,999 images!
allImageFiles = [path_of_input + '/' + f for f in listdir(path_of_input) if isfile(join(path_of_input, f))]

def weightedAverage(pixel):
    return (WEIGHT1*pixel[0] + WEIGHT2*pixel[1] + WEIGHT3*pixel[2]) / 255.0
    
def weightedAverage3Channels(pixel):
    weight = (WEIGHT1*pixel[0] + WEIGHT2*pixel[1] + WEIGHT3*pixel[2]) / 255.0
    return [weight, weight, weight]
    
final_array = np.empty((len(allImageFiles), SCALE_AFTER_RESIZE**2))
for i, file in enumerate(allImageFiles):
    if i % 100 == 0:
        print(i)
    temp = imread(file)
    temp = np.apply_along_axis(weightedAverage, 2, temp)
    temp = misc.imresize(temp, (SCALE_AFTER_RESIZE, SCALE_AFTER_RESIZE), 'nearest').flatten() / 255.0
    final_array[i, :] = temp
        
allImageNames = [n.split('/')[-1].split('.')[0] for n in allImageFiles]
final_df = pd.DataFrame(final_array)
final_df['image'] = allImageNames

final_df.to_csv('all_image_df.csv', index = False)



















