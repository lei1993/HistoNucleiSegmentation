from keras.models import load_model
from losses import bce_dice_loss,dice_coef,weighted_bce_dice_loss,dice_loss
from skimage.transform import resize
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import time
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
threshold = 0.5
random.seed = seed
np.random.seed = seed

dataset = 'TNBC'
npy_data_path = '/media/zhaolei/Data/NucleiSegmentation/npy_data'
X_npy_data = os.path.join(npy_data_path,'X_test_%s.npy'%dataset)
Y_npy_data = os.path.join(npy_data_path,'Y_test_%s.npy'%dataset)

X_test = np.load(X_npy_data)
Y_test = np.load(Y_npy_data)

model_path = '/media/zhaolei/Data/NucleiSegmentation/model'
model_prefix = 'model_weights_%s_weighted_bce_dice_loss.hdf5'%dataset
# model_prefix = 'model_weights_%s_dice_loss.hdf5'%dataset

model_name = os.path.join(model_path,model_prefix)

loss = dice_loss
model = load_model(model_name, custom_objects={'weighted_bce_dice_loss':loss,'dice_coef':dice_coef})
# model = load_model(model_name, custom_objects={'dice_loss':loss,'dice_coef':dice_coef})

t1 = time.time()
preds_test = model.predict(X_test, batch_size= 5,verbose=1)
t2 = time.time()
print t2 -t1
# Threshold predictions

preds_test_t = (preds_test > threshold).astype(np.uint8)

print preds_test_t.shape

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (512, 512),
                                       mode='constant', preserve_range=True))

pre_dicted_npy_path = '/media/zhaolei/Data/NucleiSegmentation/predict_npy'

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_test_t))
plt.figure()
plt.imshow(X_test[9])
plt.figure()
plt.imshow(np.squeeze(preds_test[9]),cmap=plt.cm.jet)
npy_name = os.path.join(pre_dicted_npy_path,'Y_pre_%s_%d_dice_loss.npy'%(dataset,threshold))
np.save(npy_name,preds_test_t)
plt.axis('off')
plt.show()