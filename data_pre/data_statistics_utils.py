import configparser
import numpy as np  # numpy数据处理
import nibabel as nib  # nii数据处理库
from numpy.ma import indices
from sklearn.preprocessing import MinMaxScaler  # 正规化
import os
import matplotlib.pyplot as plt  # 图标显示

scaler = MinMaxScaler()
conf = configparser.ConfigParser()
conf.read('/home/csunix/ml20r2w/Documents/FinalProject/conf.ini')
# 数据集路径
DATASET_DIR = conf.get('uni', 'DATASET_DIR')
TRAIN_DATASET_FOLDER_NAME = conf.get('uni', 'TRAIN_DATASET_FOLDER_NAME')
VALIDATION_DATASET_FOLDER_NAME = conf.get('uni', 'VALIDATION_DATASET_FOLDER_NAME')


def cal_peak_of_img():
    dir_list = os.listdir(DATASET_DIR + TRAIN_DATASET_FOLDER_NAME)[0:]
    dir_list.sort()

    all_statis = []

    cube_statis = np.zeros((240, 240, 155), dtype=np.uint)

    for index, item in enumerate(dir_list):
        # 读取
        FILE_INDEX = item.split('_')[1]
        SINGLE_FOLDER = f'BraTS2021_{FILE_INDEX}/BraTS2021_{FILE_INDEX}'
        print(f'处理:{FILE_INDEX},{index + 1}/{len(dir_list)}')
        # segmentation
        test_mask = nib.load(DATASET_DIR + TRAIN_DATASET_FOLDER_NAME + SINGLE_FOLDER + '_seg.nii.gz').get_fdata()
        test_mask = test_mask.astype(np.uint8)
        # 统计
        statis_nonzero = np.nonzero(test_mask)

        cube_statis[statis_nonzero] += 1

        x_start, x_end = np.min(statis_nonzero[0]), np.max(statis_nonzero[0])
        y_start, y_end = np.min(statis_nonzero[1]), np.max(statis_nonzero[1])
        z_start, z_end = np.min(statis_nonzero[2]), np.max(statis_nonzero[2])

        all_statis.append([
            x_start,
            x_end,
            y_start,
            y_end,
            z_start,
            z_end,
        ])

    all_statis = np.array(all_statis)

    x_start = np.min(all_statis[:, 0])
    x_end = np.max(all_statis[:, 1])
    y_start = np.min(all_statis[:, 2])
    y_end = np.max(all_statis[:, 3])
    z_start = np.min(all_statis[:, 4])
    z_end = np.max(all_statis[:, 5])

    cube_x = [cube_statis[x, :, :] for x in range(cube_statis.shape[0])]
    cube_y = [cube_statis[:, y, :] for y in range(cube_statis.shape[1])]
    cube_z = [cube_statis[:, :, z] for z in range(cube_statis.shape[2])]

    return [x_start, x_end, y_start, y_end, z_start, z_end, all_statis, cube_statis, cube_x, cube_y, cube_z, ]


result = cal_peak_of_img()
result_T_arr = result[6].T

# ===============================================================
# 可视化
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.hist(result_T_arr[0], bins=[x for x in range(240)])
plt.title("X_start")

plt.subplot(234)
plt.hist(result_T_arr[1], bins=[x for x in range(240)])
plt.title("X_end")

plt.subplot(232)
plt.hist(result_T_arr[2], bins=[x for x in range(240)])
plt.title("Y_start")

plt.subplot(235)
plt.hist(result_T_arr[3], bins=[x for x in range(240)])
plt.title("Y_end")

plt.subplot(233)
plt.hist(result_T_arr[4], bins=[x for x in range(155)])
plt.title("Z_start")

plt.subplot(236)
plt.hist(result_T_arr[5], bins=[x for x in range(155)])
plt.title("Z_end")

plt.show()
# ===============================================================

# ===============================================================
# 可视化 统计所有图像中，mask主要覆盖的区域
plt.figure(figsize=(12, 4))

plt.subplot(131)
x = np.array([np.sum(x) for x in result[8]], dtype=int)
x_p = np.percentile(x, 46.3)
x[x < x_p] = 0
plt.plot(x, marker=',', linewidth=1)
plt.xlabel(
    f'min={np.min(np.nonzero(x)[0])} max={np.max(np.nonzero(x)[0])} gap={np.max(np.nonzero(x)[0]) - np.min(np.nonzero(x)[0])}')
plt.title("Sagittal")

plt.subplot(132)
y = np.array([np.sum(y) for y in result[9]], dtype=int)
y_p = np.percentile(y, 46.3)
y[y < y_p] = 0
plt.plot(y, marker=',', linewidth=1)
plt.xlabel(
    f'min={np.min(np.nonzero(y)[0])} max={np.max(np.nonzero(y)[0])} gap={np.max(np.nonzero(y)[0]) - np.min(np.nonzero(y)[0])}')
plt.title("Coronal")

plt.subplot(133)
z = np.array([np.sum(z) for z in result[10]], dtype=int)
z_p = np.percentile(z, 16.5)
z[z < z_p] = 0
plt.plot(z, marker=',', linewidth=1)
plt.xlabel(
    f'min={np.min(np.nonzero(z)[0])} max={np.max(np.nonzero(z)[0])} gap={np.max(np.nonzero(z)[0]) - np.min(np.nonzero(z)[0])}')
plt.title("Axial")

plt.show()
# ===============================================================

# TODO 将上面的图片保存为svg写入论文
