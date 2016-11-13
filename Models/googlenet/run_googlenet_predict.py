import numpy as np
import pandas as pd

import sys, os, caffe
from os import listdir
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + './deploy.prototxt'
model_weights = caffe_root + './googlenet_iter_3000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('./ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

test_folder = '../../data/roof_augmented/test/'
allTestFiles = [f for f in listdir(test_folder)]

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

def computeProb(img):
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = img

    ### perform classification
    output = net.forward()
    return output['prob']

preds = []
for f in allTestFiles:    

    image = caffe.io.load_image(test_folder+f)
    transformed_image = transformer.preprocess('data', image)
    p = computeProb(transformed_image)
    preds.append(p.argmax())


df = pd.DataFrame(preds)
df['Id'] = pd.Series([f.split('.')[0] for f in allTestFiles])
df.rename(columns={0:'label'}, inplace=True)
df[['Id', 'label']].to_csv('googlenet_preds.csv', index=False)
