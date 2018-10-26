# -*- coding=utf-8 -*-
'''
    Eval on all data and calculate mean and standard devitation for
    mean IOU, Pixel Acc, mean ACC, dice coefficient
'''
# loading all required packages
from eval_segm import mean_IU,pixel_accuracy,mean_accuracy
from skimage.transform import resize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import warnings
import numpy as np
import csv
import os
warnings.filterwarnings('ignore')
smooth = 1
threshold = 0.8
dataset = 'multiorgan'
csv_path = '/media/zhaolei/Data/NucleiSegmentation/Eval'
csv_name = os.path.join(csv_path,'dice_loss_%s_%.1f.csv'%(dataset,threshold))
f = open(csv_name,'w')
csv_writer = csv.writer(f)
headers = ['mean IOU','Pixel Acc','mean ACC','Dice']
csv_writer.writerow(headers)
Boxplot_path = '/media/zhaolei/Data/NucleiSegmentation/Boxplot'


# define metrics
def dice_coef(y_true, y_pred):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


Y_true_all_name = '/media/zhaolei/Data/NucleiSegmentation/npy_data/Y_test_%s.npy'%dataset
Y_pre_all_name = '/media/zhaolei/Data/NucleiSegmentation/predict_npy/Y_pre_%s_%d_dice_loss.npy'%(dataset,threshold)
Y_true_all = np.load(Y_true_all_name)
Y_pre_all = np.load(Y_pre_all_name)

print Y_true_all.shape
print Y_pre_all.shape

mean_IOU_pool = []
Pixel_ACC_pool = []
Mean_ACC_pool = []
Dice_pool = []

for i in range(0,Y_true_all.shape[0]):
    # Define IoU metric
    y_true = Y_true_all[i,:,:,:1]
    y_true = np.squeeze(y_true)
    y_true = resize(y_true, (512, 512), mode='constant',preserve_range=True)
    thresh = threshold_otsu(y_true)
    y_true = y_true > thresh
    # y_true = y_true / 255
    y_pred = Y_pre_all[i,:,:,:1]
    y_pred = np.squeeze(y_pred)

    print y_true.shape,y_pred.shape

    meanIOU = mean_IU(y_pred,y_true)
    print "meanIOU:"
    print meanIOU
    ac = pixel_accuracy(y_pred, y_true)
    print 'Pixel ACC:'
    print ac

    mean_acc = mean_accuracy(y_pred, y_true)
    print "Mean ACC:"
    print mean_acc

    dice = dice_coef(y_pred, y_true)
    print "Dice:"
    print dice
    write_data = [str(meanIOU),str(ac),str(mean_acc),str(dice)]
    csv_writer.writerow(write_data)
    mean_IOU_pool.append(meanIOU)
    Pixel_ACC_pool.append(ac)
    Mean_ACC_pool.append(mean_acc)

    Dice_pool.append(dice)

IOU_std = np.std(mean_IOU_pool)
IOU_mean = np.mean(mean_IOU_pool)

Pixel_ACC_std = np.std(Pixel_ACC_pool)
Pixel_ACC_mean = np.mean(Pixel_ACC_pool)

Mean_ACC_std = np.std(Mean_ACC_pool)
Mean_ACC_mean = np.mean(Mean_ACC_pool)

dice_std = np.std(Dice_pool)
dice_mean = np.mean(Dice_pool)
print "IOU standard Deviation:"
print IOU_std

print "IOU mean:"
print IOU_mean

print "Pixel ACC standard Deviation:"
print Pixel_ACC_std

print "Pixel ACC mean:"
print Pixel_ACC_mean

print "Mean ACC standard Deviation:"
print Mean_ACC_std

print "Mean ACC mean:"
print Mean_ACC_mean

print "dice standard Deviation:"
print dice_std

print "dice mean:"
print dice_mean

plt.figure()
plt.boxplot(Dice_pool)
plt.title(dataset)
plt.savefig('%s/BoxPlot_%s_%d_dice_loss.png'%(Boxplot_path,dataset,threshold),dpi=300)
plt.show()
f.close()