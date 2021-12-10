import sys

sys.path.append("/home/csunix/ml20r2w/Documents/FinalProject")
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

# 确保使用GPU
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
print(physical_gpus)
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import model.simple_3d_unet as SimpleUnet
from data_pre.data_generator import image_loader
from model.loss_function import AuxLossFunction
from model.metric_function import DiceScore

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

# 簇数量
batch_size = 2

train_img_data_generator = image_loader(train_img_dir, train_img_list,
                                        train_mask_dir, train_mask_list, batch_size)

val_img_data_generator = image_loader(val_img_dir, val_img_list,
                                      val_mask_dir, val_mask_list, batch_size)

########################################################################
# PART 4
# 设置损失函数，优化器，学习率
# Define loss, metrics and optimizer to be used for training
# 使用混合精度提高性能
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)
dt = tf.float32

wt0, wt1, wt2, wt3 = (0.26, 33.8, 33.8, 24.87)
max_epoch = 100

total_loss = [AuxLossFunction(sample_weight=[wt0, wt1, wt2, wt3])]

init_lr = 0.0005


def scheduler(epoch, lr):
    step_1_max_lr = 0.0080
    step_1_amount = 0.0005
    if epoch <= step_1_max_lr / step_1_amount:
        # warmup
        return lr + step_1_amount
    else:
        # poly learning rate
        return math.pow((1 - epoch / max_epoch), 0.9) / 100


lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

optim = tf.keras.optimizers.SGD(learning_rate=init_lr, momentum=0.9)

# metrics = [DiceScore()]
# metrics = [DiceScore(sample_weight=[wt0, wt1, wt2, wt3], dtype=dt), Hausdorff95Score(dtype=dt)]
metrics = [DiceScore(sample_weight=[wt0, wt1, wt2, wt3], dtype=dt)]

run_eagerly = True

########################################################################
# PART 5
# save model

t = time.localtime()

# Uni path
model_folder = f"{uni_path}/save_model"
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

model_type = "SimpleUnet"
model_create_time = f"{t.tm_year}{t.tm_mon}{t.tm_mday}{t.tm_hour}{t.tm_min}"
model_sample_count = len(train_img_list)
model_batch = batch_size
model_epoch = max_epoch
model_extention = ".hdf5"
model_checkpoints_path = "checkpoints"
# Set save info
model_name = f"model_{model_type}_{model_create_time}_{model_sample_count}sample_{model_batch}batch_{model_epoch}epoch"
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

# 加载模型
model = SimpleUnet.simple_unet_model()
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
