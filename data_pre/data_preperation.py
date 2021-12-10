"""
数据准备内容

引用 https://youtu.be/oB35sV1npVI 的内容：

https://pypi.org/project/nibabel/

All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings

T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

Annotations comprise
- the GD-enhancing tumor (ET — label 4) 增强的肿瘤
- the peritumoral edematous/invaded tissue (ED — label 2) 坏死的核心
- the necrotic tumor core (NCR — label 1) 水肿
- nothing (label 0) 正常部分
"""

import configparser
import numpy as np  # numpy数据处理
import nibabel as nib  # nii数据处理库
from sklearn.preprocessing import MinMaxScaler  # 正规化
import matplotlib.pyplot as plt  # 图标显示
from tifffile import imsave  # 图像保存
from tensorflow.keras.utils import to_categorical  # 图像加载

scaler = MinMaxScaler()
conf = configparser.ConfigParser()
conf.read('conf.ini')
# 数据集路径
DATASET_DIR = conf.get('path', 'DATASET_DIR')
TRAIN_DATASET_FOLDER_NAME = conf.get('path', 'TRAIN_DATASET_FOLDER_NAME')
VALIDATION_DATASET_FOLDER_NAME = conf.get('path', 'VALIDATION_DATASET_FOLDER_NAME')

# 读取
FILE_INDEX = '00018'
SINGLE_FOLDER = f'BraTS2021_{FILE_INDEX}/BraTS2021_{FILE_INDEX}'
# flair
test_image_flair = nib.load(DATASET_DIR + TRAIN_DATASET_FOLDER_NAME + SINGLE_FOLDER + '_flair.nii.gz').get_fdata()
print(test_image_flair.max())
# 先缩放为1D，然后再缩放回去，以适配scaler函数
test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(
    test_image_flair.shape)
# t1
test_image_t1 = nib.load(DATASET_DIR + TRAIN_DATASET_FOLDER_NAME + SINGLE_FOLDER + '_t1.nii.gz').get_fdata()
test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)
# t1ce
test_image_t1ce = nib.load(DATASET_DIR + TRAIN_DATASET_FOLDER_NAME + SINGLE_FOLDER + '_t1ce.nii.gz').get_fdata()
test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(
    test_image_t1ce.shape)
# t2
test_image_t2 = nib.load(DATASET_DIR + TRAIN_DATASET_FOLDER_NAME + SINGLE_FOLDER + '_t2.nii.gz').get_fdata()
test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)
# segmentation
test_mask = nib.load(DATASET_DIR + TRAIN_DATASET_FOLDER_NAME + SINGLE_FOLDER + '_seg.nii.gz').get_fdata()
test_mask = test_mask.astype(np.uint8)

# 将标签 0124 改为 0123 以适配后面模型的参数 训练结束后 再转化出来.
# print(np.unique(test_mask))
test_mask[test_mask == 4] = 3
# print(np.unique(test_mask))

# 实际使用的标签只占用总体积的不到 百分之一
# print(np.count_nonzero(test_mask))
# print(57305/8928000)

# ===============================================================
# 可视化
import random

# 选择切片以及显示范围
pix_start = 50
pix_end = 190
n_slice = random.randint(50, test_mask.shape[2] - 50)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.imshow(test_image_flair[pix_start:pix_end, pix_start:pix_end, n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[pix_start:pix_end, pix_start:pix_end, n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1ce[pix_start:pix_end, pix_start:pix_end, n_slice], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_image_t2[pix_start:pix_end, pix_start:pix_end, n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[pix_start:pix_end, pix_start:pix_end, n_slice], cmap='gray')
plt.title('Mask')
plt.show()
# ===============================================================


# 将4个3D图像合并为一个图像，创建4通道的多模态图像(240, 240, 155, 4)
# 新图像分别为：x*y*z*{0,1,2,3}
# 0     1     2     3
# T1    T1CE  T2    FLAIR
combined_img = np.stack([test_image_t1, test_image_t1ce, test_image_t2, test_image_flair], axis=3)

# 为了节约算力，将数据进行裁剪为正方体，大小为2的平方数，用来适配GPU
crop_step = 128
combined_img = combined_img[55:55 + crop_step, 63:63 + crop_step, 12:12 + crop_step]  # Crop to 128x128x128x4
# 训练时可将128分割为多个patches
# 同时分割seg部分
test_mask = test_mask[55:55 + crop_step, 63:63 + crop_step, 12:12 + crop_step]

# ===============================================================
# 可视化
# 选择切片以及显示范围
pix_start = 0
pix_end = 128
n_slice = random.randint(50, test_mask.shape[2] - 50)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.imshow(combined_img[pix_start:pix_end, pix_start:pix_end, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(combined_img[pix_start:pix_end, pix_start:pix_end, n_slice, 1], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(combined_img[pix_start:pix_end, pix_start:pix_end, n_slice, 2], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(combined_img[pix_start:pix_end, pix_start:pix_end, n_slice, 3], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[pix_start:pix_end, pix_start:pix_end, n_slice], cmap='gray')
plt.title('Mask')
plt.show()
# ===============================================================

imsave('combined128.tif', combined_img)
np.save('combined128.npy', combined_img)

my_img = np.load('combined128.npy')

# 此步骤转之为热编码
'''
转换前
test_mask[81,94,50]
Out[7]: 0
test_mask[81,94,60]
Out[8]: 2
test_mask[81,94,70]
Out[9]: 3
转换后
test_mask[81,94,70]
Out[11]: array([0., 0., 0., 1.], dtype=float32)
test_mask[81,94,60]
Out[12]: array([0., 0., 1., 0.], dtype=float32)
test_mask[81,94,50]
Out[13]: array([1., 0., 0., 0.], dtype=float32)
'''
test_mask = to_categorical(test_mask, num_classes=4)
