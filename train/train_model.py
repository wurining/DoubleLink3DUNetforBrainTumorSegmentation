"""
@Create by Rining Wu
@Email ml20r2w@leeds.ac.uk

1. 输入 128*128*128*4通道，根据实际计算资源，在生成过程中对图像大小进行更改。
  - 将4个3D图像合并为一个图像，创建4通道的多模态图像(240, 240, 155, 4)
  - 新图像分别为：x*y*z*{0,1,2,3}
  - 0     1     2     3
  - T1    T1CE  T2    FLAIR

2.
"""
import configparser
import os
import numpy as np
from data_pre.data_generator import image_loader
from matplotlib import pyplot as plt
import glob
import random

####################################################
# PART 1

# 加载配置
conf = configparser.ConfigParser()
conf.read('conf.ini')
# 数据集路径
DATASET_DIR = conf.get('path', 'DATASET_DIR')
TRAIN_DATASET_FOLDER_NAME = conf.get('path', 'TRAIN_DATASET_FOLDER_NAME')
VALIDATION_DATASET_FOLDER_NAME = conf.get('path', 'VALIDATION_DATASET_FOLDER_NAME')
# 合并后的数据集path
train_img_dir = f"{DATASET_DIR}/combined_data/images/"
train_mask_dir = f"{DATASET_DIR}/combined_data/masks/"
# 加载文件列表
img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)
# 数据集数量
num_images = len(os.listdir(train_img_dir))

#############################################################
# PART 2
# TODO 测试加载是否成功
img_num = random.randint(0, num_images - 1)

test_img = np.load(train_img_dir + img_list[img_num])
test_mask = np.load(train_mask_dir + msk_list[img_num])
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(233)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(234)
plt.imshow(test_img[:, :, n_slice, 3], cmap='gray')
plt.title('Image t1')
plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

#############################################################
# PART 3 计算数据集中每个分类的大约权重
# 该步骤为了改善"类不平衡"问题？可以尝试计算，然后看模型表现
# Optional step of finding the distribution of each class and calculating appropriate weights
# Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

import pandas as pd

train_mask_list = sorted(glob.glob(f"{train_mask_dir}*.npy"))
# 创建df用于统计
columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)
# 计算各个分类的数量
for img in range(len(train_mask_list)):
    # 加载数据
    temp_image = np.load(train_mask_list[img])
    # 从One-Hot code解码
    temp_image = np.argmax(temp_image, axis=3)
    # 计算每个分类的个数
    val, counts = np.unique(temp_image, return_counts=True)
    # 打包为字典，适配data frame
    zipped = zip(columns, counts)
    conts_dict = dict(zipped)
    # 加入DF
    df = df.append(conts_dict, ignore_index=True)

# 计算类权重
label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['1'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4
# Class weights calculation: n_samples / (n_classes * n_samples_for_class)
# [How to Improve Class Imbalance using Class Weights in Machine Learning](https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/)
wt0 = round((total_labels / (n_classes * label_0)), 2)
wt1 = round((total_labels / (n_classes * label_1)), 2)
wt2 = round((total_labels / (n_classes * label_2)), 2)
wt3 = round((total_labels / (n_classes * label_3)), 2)

# Weights 10file are: 0.26, 22.14, 22.14, 39.14
# wt0, wt1, wt2, wt3 = 0.26, 22.14, 22.14, 39.14
# These weihts can be used for Dice loss

##############################################################
# PART 4
# 数据集Path
# 合并后的数据集path
train_img_dir = f"{DATASET_DIR}/split_combined_data/train/images/"
train_mask_dir = f"{DATASET_DIR}/split_combined_data/train/masks/"

val_img_dir = f"{DATASET_DIR}/split_combined_data/val/images/"
val_mask_dir = f"{DATASET_DIR}/split_combined_data/val/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

########################################################################
# PART 5
# 设置数据加载器

# 簇数量
batch_size = 2

train_img_data_generator = image_loader(train_img_dir, train_img_list,
                                        train_mask_dir, train_mask_list, batch_size)

val_img_data_generator = image_loader(val_img_dir, val_img_list,
                                      val_mask_dir, val_mask_list, batch_size)

# 测试加载
img, msk = next(train_img_data_generator)

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(233)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(234)
plt.imshow(test_img[:, :, n_slice, 3], cmap='gray')
plt.title('Image t1')
plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

########################################################################
# PART 6
# 设置损失函数，优化器，学习率
# Define loss, metrics and optimizer to be used for training
wt0, wt1, wt2, wt3 = 0.26, 22.14, 22.14, 39.14
import segmentation_models_3D as sm
import tensorflow as tf

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)
########################################################################
# PART 6
# 训练模型
# Fit the model

# 地板除得出epoch数量
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

# 加载模型
from model.simple_3d_unet import simple_unet_model

model = simple_unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=4,
                          num_classes=4)
# 编译模型
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)
# 输出结果
history = model.fit(train_img_data_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    validation_data=val_img_data_generator,
                    validation_steps=val_steps_per_epoch,
                    )
# 模型保存
model.save('brats_3d.hdf5')
##################################################################


# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

########################################################################
# PART 7 继续训练模型
from keras.models import load_model

# Load model for prediction or continue training

# For continuing training....
# The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
# This is because the model does not save loss function and metrics. So to compile and
# continue training we need to provide these as custom_objects.
my_model = load_model('brats_3d.hdf5')

# So let us add the loss as custom object... but the following throws another error...
# Unknown metric function: iou_score
my_model = load_model('brats_3d.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss})

# Now, let us add the iou_score function we used during our initial training
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score': sm.metrics.IOUScore(threshold=0.5)})

# Now all set to continue the training process.
history2 = my_model.fit(train_img_data_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=1,
                        verbose=1,
                        validation_data=val_img_data_generator,
                        validation_steps=val_steps_per_epoch,
                        )
