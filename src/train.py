import numpy as np
import matplotlib.pyplot as plt
import Augmentor
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard
from aug import get_more_images,duplicate_labels
from model import get_dilated_unet
import os
WIDTH = 512
HEIGHT = 512
BATCH_SIZE = 2
dataset = 'alldataraw'
npy_data_path = '/media/zhaolei/Data/NucleiSegmentation/npy_data'
X_npy_data = os.path.join(npy_data_path,'X_train_%s.npy'%dataset)
Y_npy_data = os.path.join(npy_data_path,'Y_train_%s.npy'%dataset)

X_train = np.load(X_npy_data)
Y_train = np.load(Y_npy_data)

X_validation = X_train[-5:,...]
Y_validation = Y_train[-5:]
X_train = X_train[0:-5,...]
Y_train = Y_train[0:-5]
print("Validation Set of Size "+str(Y_validation.shape[0])+" Separated")
Y_train.dtype = np.uint8
X_train = get_more_images(X_train)
Y_train = duplicate_labels(Y_train)
print (X_train.shape)
print (Y_train.shape)
print (X_validation.shape)
print (Y_validation.shape)
print("Data Rotation and Flipping Complete")
print (X_train.shape)
print (Y_train.shape)
print (X_validation.shape)
print (Y_validation.shape)

p = Augmentor.Pipeline()
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
g = p.keras_generator_from_array(X_train, Y_train, batch_size=20)
p.status()
model_path = '/media/zhaolei/Data/NucleiSegmentation/model'
# model_prefix = 'model_weights_%s_weighted_bce_dice_loss.hdf5'%dataset
model_prefix = 'model_weights_%s_dice_loss.hdf5'%dataset

model_name = os.path.join(model_path,model_prefix)
logdir = '/media/zhaolei/Data/NucleiSegmentation/log'
log_name = os.path.join(logdir,'dice_loss_%s_log.txt'%(dataset))
f = open(log_name,'w')
losscurve_path = '/media/zhaolei/Data/NucleiSegmentation/losscurve'
if __name__ == '__main__':

    model = get_dilated_unet(
        input_shape=(512, 512,3),
        mode='cascade',
        filters=32,
        n_class=1
    )

    callbacks = [EarlyStopping(monitor='val_dice_coef',
                               patience=7,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_dice_coef',
                                   factor=0.2,
                                   patience=1,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_dice_coef',
                                 filepath=model_name,
                                 save_best_only=True,
                                 mode='max')]

    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=2, epochs=200,callbacks=callbacks)

    history = results
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model Dice_Coefficient')
    plt.ylabel('Dice_coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig('%s/Dice_coef_dilate_%s_weighted_bce_dice_loss.png'%(losscurve_path,dataset), dpi=300)
    plt.savefig('%s/Dice_coef_dilate_%s_dice_loss.png' % (losscurve_path, dataset), dpi=300)

    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    # plt.savefig('%s/Loss_dilate_%s_weighted_bce_dice_loss.png'%(losscurve_path,dataset), dpi=300)
    plt.savefig('%s/Loss_dilate_%s_dice_loss.png' % (losscurve_path, dataset), dpi=300)

    plt.show()

    Train_loss = str(history.history['loss']) + '\n'
    Val_loss = str(history.history['val_loss']) + '\n'
    Train_dice_coef = str(history.history['dice_coef']) + '\n'
    Val_dice_coef = str(history.history['val_dice_coef']) + '\n'
    f.write(Train_loss)
    f.write(Val_loss)
    f.write(Train_dice_coef)
    f.write(Val_dice_coef)
    f.close()



