import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#which batch and which epoch
num_round = int(sys.argv[1])
num_batch = int(sys.argv[2])
prefix = "model/incept-%d-0"%num_batch
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(1), numpy_batch_size=1)
mean_img = mx.nd.load("mean.bin")["mean_img"]#("mean%d.bin"%num_batch)["mean_img"]

def PreprocessImage(path, show_img=False):
    # load image
    img = io.imread(path)
    #print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean 
    normed_img = sample - mean_img.asnumpy()
    normed_img.resize(1, 3, 224, 224)
    return normed_img

import os
test_images_path = '../../data/test/'
test_list = sorted(os.listdir(test_images_path))
fw = open('submission_test_batch_%d_epoch_%d.csv'%(num_batch, num_round),'w')
fw.write('Id,label,label1,label2,label3,label4\n')

cnt = 0
for f in test_list:
    cnt+=1
    batch = PreprocessImage(test_images_path+'%s'%(f))
    prob = model.predict(batch)[0][1:5]
    # Argsort, get prediction index from largest prob to lowest
    pred = np.argsort(prob)[::-1][0] + 1
    #prob /= np.sum(prob)
    output_prob = ['%.6f'%p for p in prob]
    fw.write('%s,%d,%s\n'%(f.split('.')[0],pred,','.join(output_prob)))

fw.close()
