import os
from os.path import isfile, join
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import pandas as pd
from random import shuffle
import shutil

path_aug = '../../data/roof_augmented'
test_data_dir = path_aug+'/test/'
allTestFiles = [f for f in listdir(test_data_dir)]
lbl = pd.read_csv('../../data/id_train.csv', dtype={'Id':str})
train_ids_set = set(lbl['Id'].values)

# path to the model weights file.
weights_path = '../../data/pretrain/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 35997
nb_validation_samples = 11998
nb_epoch = 50

#def save_bottlebeck_features():
datagen = ImageDataGenerator(rescale=1./255)

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')


train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        classes = ['1','2','3','4'],   
        shuffle=True)
bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
#np.save(open('bottleneck_train_labels.npy', 'w'), generator.classes)

test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=32)

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        classes = ['1','2','3','4'],   
        shuffle=True)
bottleneck_features_validation = model.predict_generator(validation_generator, nb_validation_samples)
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
#np.save(open('bottleneck_validation_labels.npy', 'w'), generator.classes)   

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])    

model.fit(train_data, train_labels,
          nb_epoch=nb_epoch, batch_size=32,
          validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path, overwrite=True)
preds = model.predict(test_generator)
df = pd.DataFrame(preds)
df['Id'] = pd.Series([f.split('.')[0] for f in allTestFiles])
df.rename(columns={0:'label'}, inplace=True)
df[['Id', 'label']].to_csv('keras_vgg_preds.csv', index=False)
