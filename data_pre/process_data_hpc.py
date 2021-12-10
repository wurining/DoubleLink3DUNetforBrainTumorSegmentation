import sys

sys.path.append("/home/home02/ml20r2w/FinalProject")
# 数据集预处理
import configparser
import glob
import os.path

import nibabel as nib  # nii数据处理库
import numpy as np  # numpy数据处理
import splitfolders
from sklearn.preprocessing import MinMaxScaler  # 正规化
from tensorflow.keras.utils import to_categorical  # 图像加载

scaler = MinMaxScaler()

conf = configparser.ConfigParser()
conf.read('/home/home02/ml20r2w/FinalProject/conf.ini')

running_location = 'hpc'  # 'google'
# 数据集路径
DATASET_DIR = conf.get(running_location, 'DATASET_DIR')
TRAIN_DATASET_FOLDER_NAME = conf.get(running_location, 'TRAIN_DATASET_FOLDER_NAME')
VALIDATION_DATASET_FOLDER_NAME = conf.get(running_location, 'VALIDATION_DATASET_FOLDER_NAME')

# 取得数据集路径
t1_list = sorted(glob.glob(f'{DATASET_DIR}{TRAIN_DATASET_FOLDER_NAME}*/*_t1.nii.gz'))
t1ce_list = sorted(glob.glob(f'{DATASET_DIR}{TRAIN_DATASET_FOLDER_NAME}*/*_t1ce.nii.gz'))
t2_list = sorted(glob.glob(f'{DATASET_DIR}{TRAIN_DATASET_FOLDER_NAME}*/*_t2.nii.gz'))
flair_list = sorted(glob.glob(f'{DATASET_DIR}{TRAIN_DATASET_FOLDER_NAME}*/*_flair.nii.gz'))
mask_list = sorted(glob.glob(f'{DATASET_DIR}{TRAIN_DATASET_FOLDER_NAME}*/*_seg.nii.gz'))

for (index, url) in enumerate(t1_list):  # Using t1_list as all lists are of same size
    # 设置分割数量
    #     if index > 10:
    #         break
    file_prefix = url.split('/')[-2]
    print("Now preparing image and masks number: ", index, file_prefix)

    temp_image_t1 = nib.load(t1_list[index]).get_fdata()
    temp_image_t1 = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(
        temp_image_t1.shape)

    temp_image_t2 = nib.load(t2_list[index]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
        temp_image_t2.shape)

    temp_image_t1ce = nib.load(t1ce_list[index]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(
        temp_image_t1ce.shape)

    temp_image_flair = nib.load(flair_list[index]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
        temp_image_flair.shape)

    temp_mask = nib.load(mask_list[index]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3

    temp_combined_img = np.stack([temp_image_t1, temp_image_t1ce, temp_image_t2, temp_image_flair], axis=3)

    # 为了节约算力，将数据进行裁剪为正方体，大小为2的平方数，用来适配GPU
    crop_step = 128
    temp_combined_img = temp_combined_img[55:55 + crop_step, 63:63 + crop_step,
                        12:12 + crop_step]  # Crop to 128x128x128x4
    # 训练时可将128分割为多个patches
    # 同时分割seg部分
    temp_mask = temp_mask[55:55 + crop_step, 63:63 + crop_step, 12:12 + crop_step]

    val, counts = np.unique(temp_mask, return_counts=True)

    temp_mask = to_categorical(temp_mask, num_classes=4)

    # 文件保存
    if not os.path.exists(f'{DATASET_DIR}combined_data/'):
        os.makedirs(f'{DATASET_DIR}combined_data/')
        os.makedirs(f'{DATASET_DIR}combined_data/images/')
        os.makedirs(f'{DATASET_DIR}combined_data/masks/')
    np.save(f'{DATASET_DIR}combined_data/images/image_' + str(file_prefix) + '.npy', temp_combined_img)
    print(f'combine image {file_prefix} done.')
    np.save(f'{DATASET_DIR}combined_data/masks/mask_' + str(file_prefix) + '.npy', temp_mask)
    print(f'combine mask {file_prefix} done.')

# 数据集分割
# TODO 将训练数据分开2/8 用于10次交叉验证


print('=================Start Copy=================')
input_folder = f'{DATASET_DIR}combined_data/'
output_folder = f'{DATASET_DIR}split_combined_data/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=1, ratio=(.8, .2), group_prefix=None)
# default values
print('=================End Copy=================')
