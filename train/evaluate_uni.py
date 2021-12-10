import sys

sys.path.append("/home/csunix/ml20r2w/Documents/FinalProject")
"""
@Create by Rining Wu
@Email ml20r2w@leeds.ac.uk

"""
import math
import configparser
import time
import datetime
import os
import tensorflow as tf
import numpy as np

# 确保使用GPU
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
print(physical_gpus)
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

from data_pre.data_generator import image_loader
from model.loss_function import AuxLossFunction, MainLossFunction
from model.metric_function import DiceScore, Hausdorff95Score, Sensitivity, Specificity

####################################################
# PART 1
uni_path = "/home/csunix/ml20r2w/Documents/not-backed-up"

conf = configparser.ConfigParser()
conf.read('/home/csunix/ml20r2w/Documents/FinalProject/conf.ini')

running_location = 'uni'
DATASET_DIR = conf.get(running_location, 'DATASET_DIR')
TRAIN_DATASET_FOLDER_NAME = conf.get(running_location, 'TRAIN_DATASET_FOLDER_NAME')
VALIDATION_DATASET_FOLDER_NAME = conf.get(running_location, 'VALIDATION_DATASET_FOLDER_NAME')
##############################################################
# PART 2
# 数据集Path
# 合并后的数据集path
train_img_dir = f"{DATASET_DIR}/split_combined_data/train/images/"
train_mask_dir = f"{DATASET_DIR}/split_combined_data/train/masks/"

val_img_dir = f"{DATASET_DIR}/split_combined_data/val/images/"
val_mask_dir = f"{DATASET_DIR}/split_combined_data/val/masks/"

train_size = 1000 // 4
val_size = 250 // 4
train_img_list = os.listdir(train_img_dir)[train_size:]
train_mask_list = os.listdir(train_mask_dir)[train_size:]

val_img_list = os.listdir(val_img_dir)[val_size:]
val_mask_list = os.listdir(val_mask_dir)[val_size:]

########################################################################
# PART 3
# 设置数据加载器

train_img_data_generator = image_loader(train_img_dir, train_img_list,
                                        train_mask_dir, train_mask_list, batch_size=1)

val_img_data_generator = image_loader(val_img_dir, val_img_list,
                                      val_mask_dir, val_mask_list, batch_size=1)

########################################################################
# PART 6
# 评价模型
# Evaluation the model

from model.simple_3d_unet import SimpleUnetModel
from model.modality_pairing_net import ModalityPairingNetTrain, AuxLossLayer, ModalityPairingLossLayer
from model.modality_pairing_net_mid import ModalityPairingNetTrain, AuxLossLayer, ModalityPairingLossLayer
import pandas as pd

# Group A
my_model = tf.keras.models.load_model(
    '/home/csunix/ml20r2w/Documents/not-backed-up/save_model/model_MPN_2021126414_96_250sample_1batch_200epoch/model_MPN_2021126414_96_250sample_1batch_200epoch.hdf5',
    compile=False,
    custom_objects={'ModalityPairingNetTrain': ModalityPairingNetTrain,
                    'AuxLossLayer': AuxLossLayer,
                    'ModalityPairingLossLayer': ModalityPairingLossLayer})

# Group B
# my_model = tf.keras.models.load_model(
#     '/home/csunix/ml20r2w/Documents/not-backed-up/save_model/model_MPN_20211242122_43_250sample_2batch_200epoch Group B Perfect/model_MPN_20211242122_43_250sample_2batch_200epoch.hdf5',
#     compile=False,
#     custom_objects={'ModalityPairingNetTrain': ModalityPairingNetTrain,
#                     'AuxLossLayer': AuxLossLayer,
#                     'ModalityPairingLossLayer': ModalityPairingLossLayer})

# Group C
# my_model = tf.keras.models.load_model(
#     '/usr/not-backed-up/ml20r2w/save_model/model_SimpleUnet_2021113129_250sample_2batch_100epoch/model_SimpleUnet_2021113129_250sample_2batch_100epoch.hdf5',
#     compile=False,
#     custom_objects={'SimpleUnetModel': SimpleUnetModel})


results = []

# 地板除得出epoch数量
batch_size = 1
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

dice_score = DiceScore(data_type=tf.double)
sensitivity = Sensitivity(data_type=tf.double)
specificity = Specificity(data_type=tf.double)

for i in range(steps_per_epoch):
    test_image_batch, test_mask_batch = train_img_data_generator.__next__()

    tf.config.run_functions_eagerly(True)
    # test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
    # test_pred_batch = my_model.predict(test_image_batch)
    test_pred_batch = my_model.predict([test_image_batch, test_mask_batch])
    # 这里这个argmax返回的是 axis=4（这个是返回的batch的shape中，(batch_size,h,w,d,4个通道分类的one hot编码)），然后返回0123是哪个通道有值，就是哪个分类。
    # test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

    print('test_image_batch', test_image_batch.shape)
    print('test_mask_batch', test_mask_batch.shape)
    dice_score.update_state(test_mask_batch, test_pred_batch)
    sensitivity.update_state(test_mask_batch, test_pred_batch)
    specificity.update_state(test_mask_batch, test_pred_batch)

    print("dice_score =", dice_score.result().numpy())
    print("sensitivity =", sensitivity.result().numpy())
    print("specificity =", specificity.result().numpy())
    results.append(np.concatenate([
        dice_score.result().numpy(),
        sensitivity.result().numpy(),
        specificity.result().numpy(),
    ]))

print(results)
data_frame = pd.DataFrame(results,
                          columns=['dice_score_total', 'dice_score NCR', 'dice_score ED', 'dice_score ET',
                                   'sensitivity_total', 'sensitivity NCR', 'sensitivity ED', 'sensitivity ET',
                                   'specificity_total', 'specificity NCR', 'specificity ED', 'specificity ET'])
print(data_frame)
print('mean axis = 1', data_frame.mean(axis=1))
data_frame.to_csv('evaluation.csv', index=False)

# ************************************************************************
# 验证输出


# n = 0
# for i in range(n):
#     train_img_data_generator.__next__()
# test_image_batch, test_mask_batch = train_img_data_generator.__next__()
# tf.config.run_functions_eagerly(True)
# test_prediction = my_model.predict(test_image_batch)
# # test_prediction = my_model.predict([test_image_batch, test_mask_batch])
# test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]
# test_mask_argmax = np.argmax(test_mask_batch[0], axis=3)
#
# # print(test_prediction_argmax.shape)
# # print(test_mask_argmax.shape)
# # print(np.unique(test_prediction_argmax))
#
#
# # Plot individual slices from test predictions for verification
# from matplotlib import pyplot as plt
# import random
#
# for i in range(5):
#     n_slice = random.randint(0, test_prediction_argmax.shape[2])
#     print(n_slice)
#
#     plt.figure(figsize=(12, 8))
#
#     plt.subplot(331)
#     plt.title(f'Testing Image:{n_slice}')
#     plt.imshow(test_image_batch[0][:, :, n_slice, 1], cmap='gray')
#     plt.subplot(332)
#     plt.title(f'Testing Label:{n_slice}')
#     plt.imshow(test_mask_argmax[:, :, n_slice])
#     plt.subplot(333)
#     plt.title(f'Prediction on test image:{n_slice}')
#     plt.imshow(test_prediction_argmax[:, :, n_slice])
#
#     n_slice = random.randint(0, test_prediction_argmax.shape[2])
#     print(n_slice)
#
#     plt.subplot(334)
#     plt.title(f'Testing Image:{n_slice}')
#     plt.imshow(test_image_batch[0][:, :, n_slice, 1], cmap='gray')
#     plt.subplot(335)
#     plt.title(f'Testing Label:{n_slice}')
#     plt.imshow(test_mask_argmax[:, :, n_slice])
#     plt.subplot(336)
#     plt.title(f'Prediction on test image:{n_slice}')
#     plt.imshow(test_prediction_argmax[:, :, n_slice])
#
#     n_slice = random.randint(0, test_prediction_argmax.shape[2])
#     print(n_slice)
#
#     plt.subplot(337)
#     plt.title(f'Testing Image:{n_slice}')
#     plt.imshow(test_image_batch[0][:, :, n_slice, 1], cmap='gray')
#     plt.subplot(338)
#     plt.title(f'Testing Label:{n_slice}')
#     plt.imshow(test_mask_argmax[:, :, n_slice])
#     plt.subplot(339)
#     plt.title(f'Prediction on test image:{n_slice}')
#     plt.imshow(test_prediction_argmax[:, :, n_slice])
#
#     plt.show()
#
# plt.savefig('eva.png')

# 生成指定图
# 可用数据 n 10 - 108
# 可用数据 n 11 - 69
# 可用数据 n 30 - 43

# n = 10
# n_slice = 108
# group = 'B'
# print(n_slice)
#
# for i in range(n):
#     train_img_data_generator.__next__()
# test_image_batch, test_mask_batch = train_img_data_generator.__next__()
# tf.config.run_functions_eagerly(True)
# # test_prediction = my_model.predict(test_image_batch)
# test_prediction = my_model.predict([test_image_batch, test_mask_batch])
# test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]
# test_mask_argmax = np.argmax(test_mask_batch[0], axis=3)
#
# # print(test_prediction_argmax.shape)
# # print(test_mask_argmax.shape)
# # print(np.unique(test_prediction_argmax))
#
# from matplotlib import pyplot as plt
#
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title(f't1 Channel:{n_slice}')
# plt.imshow(test_image_batch[0][:, :, n_slice, 0], cmap='gray')
# plt.subplot(232)
# plt.title(f't1ce Channel:{n_slice}')
# plt.imshow(test_image_batch[0][:, :, n_slice, 1], cmap='gray')
# plt.subplot(233)
# plt.title(f't2 Channel:{n_slice}')
# plt.imshow(test_image_batch[0][:, :, n_slice, 2], cmap='gray')
# plt.subplot(234)
# plt.title(f'flair Channel:{n_slice}')
# plt.imshow(test_image_batch[0][:, :, n_slice, 3], cmap='gray')
# plt.subplot(235)
# plt.title(f'Testing Label:{n_slice}')
# plt.imshow(test_mask_argmax[:, :, n_slice])
# plt.subplot(236)
# plt.title(f'Prediction on test image:{n_slice}')
# plt.imshow(test_prediction_argmax[:, :, n_slice])
#
# plt.show()
# plt.savefig(f'{n}-{n_slice}-{group}.png', dpi=300)
