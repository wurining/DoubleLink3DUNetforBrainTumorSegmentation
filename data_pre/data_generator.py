import numpy as np

import random


# 加载图片 添加至np数组
def load_img(img_dir_path, img_name_list):
    images = []

    for i, img_name in enumerate(img_name_list):
        if img_name.split('.')[1] == 'npy':
            image = np.load(img_dir_path + img_name)
            images.append(image)

    images = np.array(images)
    return images


# 加载Batch
def image_loader(img_dir, img_list, mask_dir, mask_list, batch_size, shuffle=False):
    list_length = len(img_list)
    list_index = list(range(list_length))

    # keras 要求的Generator需要可以无限迭代
    while True:
        batch_start = 0
        batch_end = batch_size
        if shuffle:
            random.shuffle(list_index)
        # 取单个Batch
        while batch_start < list_length:
            limit = min(batch_end, list_length)
            # list_index[batch_start:limit]
            X = load_img(img_dir, [img_list[j] for j in list_index[batch_start:limit]])
            Y = load_img(mask_dir, [mask_list[k] for k in list_index[batch_start:limit]])
            yield X, Y  # a tuple with two numpy arrays with batch_size samples
            # 下一组Batch的Cursor
            batch_start += batch_size
            batch_end += batch_size


############################################

# 测试代码
if __name__ == "__main__":
    import configparser
    import os

    # import matplotlib.pyplot as plt
    # import random

    ####################################################
    # PART 1
    uni_path = "/home/home02/ml20r2w/nobackup"

    conf = configparser.ConfigParser()
    conf.read('/home/home02/ml20r2w/FinalProject/conf.ini')

    running_location = 'hpc'
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

    train_img_list = os.listdir(train_img_dir)[:]
    train_mask_list = os.listdir(train_mask_dir)[:]

    val_img_list = os.listdir(val_img_dir)[:]
    val_mask_list = os.listdir(val_mask_dir)[:]

    print('=============================')
    print('train_img_list', len(train_img_list))
    print('train_mask_list', len(train_mask_list))
    print('val_img_list', len(val_img_list))
    print('val_mask_list', len(val_mask_list))
    print('=============================')

    # 簇数量
    batch_size = 1

    train_img_data_generator = image_loader(train_img_dir, train_img_list,
                                            train_mask_dir, train_mask_list, batch_size)

    val_img_data_generator = image_loader(val_img_dir, val_img_list,
                                          val_mask_dir, val_mask_list, batch_size)

    # 使用next()取第一次迭代
    i = 0
    while True:
        img_train, msk_train = next(train_img_data_generator)
        img_val, msk_val = next(val_img_data_generator)
        print(f'===============  {i}  =============')
        print('img_train shape', img_train.shape)
        print('msk_train shape', msk_train.shape)
        print('img_val shape', img_val.shape)
        print('msk_val shape', msk_val.shape)
        i += 1
