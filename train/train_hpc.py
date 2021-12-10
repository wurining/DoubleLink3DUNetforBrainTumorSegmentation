import sys
import random

sys.path.append("/home/home02/ml20r2w/FinalProject")
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
import math
import configparser
import time
import datetime
import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# 确保使用GPU
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
print(physical_gpus)
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # 设置两个逻辑GPU模拟多GPU训练
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                                 [tf.config.experimental.VirtualDeviceConfiguration(
#                                                                     memory_limit=1024 * 10),
#                                                                     tf.config.experimental.VirtualDeviceConfiguration(
#                                                                         memory_limit=1024 * 10)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)
# tf.keras.backend.clear_session()
# strategy = tf.distribute.MirroredStrategy()

import model.modality_pairing_net as MPN
import model.modality_pairing_net_mid as MPN_MID
from data_pre.data_generator import image_loader
from model.loss_function import MainLossFunction
from model.metric_function import DiceScore

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

train_size = 1000 // 4
val_size = 250 // 4
train_img_list = os.listdir(train_img_dir)[0:train_size:1]
train_mask_list = os.listdir(train_mask_dir)[0:train_size:1]

val_img_list = os.listdir(val_img_dir)[0:val_size:1]
val_mask_list = os.listdir(val_mask_dir)[0:val_size:1]

print('=============================')
print('train_img_list', len(train_img_list))
print('train_mask_list', len(train_mask_list))
print('val_img_list', len(val_img_list))
print('val_mask_list', len(val_mask_list))
print('=============================')
########################################################################
# PART 3
# 设置数据加载器
dt = tf.float32

# 簇数量
batch_size = 2

train_img_data_generator = image_loader(train_img_dir, train_img_list,
                                        train_mask_dir, train_mask_list, batch_size, shuffle=True)

val_img_data_generator = image_loader(val_img_dir, val_img_list,
                                      val_mask_dir, val_mask_list, batch_size, shuffle=False)

# train_dataset = tf.data.Dataset.from_generator(
#     lambda: (image_loader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)),
#     output_signature=(tf.TensorSpec(shape=(batch_size, 128, 128, 128, 4), dtype=tf.float32),
#                       tf.TensorSpec(shape=(batch_size, 128, 128, 128, 4), dtype=tf.float32)))
# test_dataset = tf.data.Dataset.from_generator(
#     lambda: (image_loader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)),
#     output_signature=(tf.TensorSpec(shape=(batch_size, 128, 128, 128, 4), dtype=tf.float32),
#                       tf.TensorSpec(shape=(batch_size, 128, 128, 128, 4), dtype=tf.float32)))
#
# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

########################################################################
# PART 4
# 设置损失函数，优化器，学习率
# Define loss, metrics and optimizer to be used for training


wt0, wt1, wt2, wt3 = (0.26, 33.8, 33.8, 24.87)
max_epoch = 200

init_lr = 0.001 / 10


def scheduler(epoch, lr):
    step_1_max_lr = 0.0100 / 10
    step_1_amount = 0.0005 / 10
    if epoch <= step_1_max_lr / step_1_amount:
        # warmup
        return lr + step_1_amount
    else:
        # return math.pow((1 - epoch / max_epoch), 0.9) / 100
        warmup_step = int(step_1_max_lr / step_1_amount)
        initial_learning_rate = step_1_max_lr
        end_learning_rate = 0.000
        decay_steps = max_epoch - warmup_step
        power = 0.9
        # poly learning rate
        step = min(epoch - warmup_step, decay_steps)
        return ((initial_learning_rate - end_learning_rate) * math.pow((1 - step / decay_steps),
                                                                       power)) + end_learning_rate


lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
# with strategy.scope():
# optim = tf.keras.optimizers.SGD(learning_rate=init_lr, momentum=0.9)
optim = 'adam'
total_loss = [MainLossFunction(sample_weight=[wt0, wt1, wt2, wt3], data_type=dt)]
metrics = [DiceScore(sample_weight=[wt0, wt1, wt2, wt3], dtype=dt), tf.keras.metrics.CategoricalCrossentropy()]
# metrics = [DiceScore(sample_weight=[wt0, wt1, wt2, wt3], dtype=dt), Hausdorff95Score(dtype=dt)]

run_eagerly = True

########################################################################
# PART 5
# save model

t = time.localtime()

# Uni path
model_folder = f"{uni_path}/save_model"
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

model_type = "MPN"
model_create_time = f"{t.tm_year}{t.tm_mon}{t.tm_mday}{t.tm_hour}{t.tm_min}"
model_sample_count = len(train_img_list)
model_batch = batch_size
model_epoch = max_epoch
model_extention = ".hdf5"
model_checkpoints_path = "checkpoints"
# Set save info
model_name = f"model_{model_type}_{model_create_time}_{random.randint(1, 100)}_{model_sample_count}sample_{model_batch}batch_{model_epoch}epoch"
# Set log
log_path = f"{model_folder}/{model_name}/"
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_callback = tf.keras.callbacks.CSVLogger(
    f"{model_folder}/{model_name}/history.csv", separator=',', append=False
)
########################################################################
# PART 6
# 训练模型
# Fit the model

# 地板除得出epoch数量
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

# with strategy.scope():
# 加载模型
# model = MPN.get_main_net(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=4, num_classes=4, data_type=dt)
model = MPN_MID.get_main_net(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=4, num_classes=4, data_type=dt)
# 编译模型
model.compile(optimizer=optim,
              loss=total_loss,
              metrics=metrics,
              run_eagerly=run_eagerly)
print(model.summary())
# (model.input_shape)
# (model.output_shape)

# Save model and checkpoint
if not os.path.exists(f"{model_folder}/{model_name}"):
    os.mkdir(f"{model_folder}/{model_name}")
if not os.path.exists(f"{model_folder}/{model_name}/{model_checkpoints_path}"):
    os.mkdir(f"{model_folder}/{model_name}/{model_checkpoints_path}")
# Include the epoch in the file name (uses `str.format`)
checkpoint_dir = f"{model_folder}/{model_name}/{model_checkpoints_path}"
checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"

# if recover from a checkpoint
# last_checkpoint_path = ""
# model.load_weights(last_checkpoint_path)


# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq=5 * steps_per_epoch)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))
try:
    # 输出结果
    history = model.fit(train_img_data_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=max_epoch,
                        verbose=1,
                        validation_data=val_img_data_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks=[lr_callback, cp_callback, tensorboard_callback, csv_callback],
                        )
finally:
    # 模型保存
    # create folder
    # models
    #  - model...
    #    - model_name
    #    - checkpoint
    #    - history
    #    - pic
    model.save(f"{model_folder}/{model_name}/{model_name}{model_extention}")
