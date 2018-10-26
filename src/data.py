# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:55:19 2018

@author: zhaolei
"""
import numpy as np
import random
from skimage.io import imread, imshow
from skimage.transform import resize
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import warnings
from skimage import transform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hed,rgb2hsv
from skimage import exposure
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
np.set_printoptions(threshold=np.nan)
seed = 42
random.seed = seed
np.random.seed = seed

root_path = '/media/zhaolei/Data/NucleiSegmentation/data'
npy_data_path = '/media/zhaolei/Data/NucleiSegmentation/npy_data'
# Data set name selection
# TODO 'TNBC','multiorgan','lung','breast','alldataraw'
dataset = 'TNBC'
# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

TRAIN_PATH = os.path.join(root_path,dataset,'train')
TEST_PATH = os.path.join(root_path,dataset,'test')

# Get train and test IDs
train_ids = next(os.walk(os.path.join(TRAIN_PATH,'img')))[2]
test_ids = next(os.walk(os.path.join(TEST_PATH,'img')))[2]
train_ids.sort()
test_ids.sort()
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,3), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)

print('\nGetting and resizing train images and masks ... \n')
sys.stdout.flush()


def color_deconv(img):
    hed = rgb2hed(img)
    h = rescale_intensity(hed[:,:,0],out_range=(0,255))
    # zdh = np.dstack((np.zeros_like(h),h,h))
    h = np.expand_dims(h,axis=-1)
    zdh = np.repeat(h,3,axis=-1)
    return zdh


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + '/img/' + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    zdh = color_deconv(img)
    X_train[n] = zdh
    mask_fname = id_.split('.')[0] + '_mask'+ '.png'
    mask = imread(os.path.join(TRAIN_PATH, 'mask/') + mask_fname,as_grey=True)
    mask = mask / 255
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
    Y_train[n] = mask


# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,3), dtype=np.uint8)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
sizes_test = []
print('\nGetting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + '/img/' + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    zdh = color_deconv(img)
    print zdh.shape
    X_test[n] = zdh
    mask_fname = id_.split('.')[0] + '_mask'+ '.png'
    mask = imread(os.path.join(TEST_PATH, 'mask/') + mask_fname,as_grey=True)
    mask = mask / 255
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
    Y_test[n] = mask


print('Done!')
print "Save Training data npy..."
np.save('%s/X_train_%s.npy'%(npy_data_path,dataset),X_train)
np.save('%s/Y_train_%s.npy'%(npy_data_path,dataset),Y_train)
np.save('%s/X_test_%s.npy'%(npy_data_path,dataset),X_test)
np.save('%s/Y_test_%s.npy'%(npy_data_path,dataset),Y_test)

plt.figure()
imshow(X_train[30])
plt.figure()
imshow(np.squeeze(Y_train[30]))
plt.show()