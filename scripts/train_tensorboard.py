import os
from glob import glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle

import time
import cv2
NAME = "Driver-Drowsiness-CNN"
channel = 3
img_size = 224
n_classes = 10
epochs = 50
batch_size = 50
test_size = 10

train_images = []
train_labels = []

# Load the dataset
def get_cv2_image(path, img_size, color_type):
    # Loading as Grayscale image
    if color_type == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Loading as color image
    elif color_type == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    # Reduce size
    img = cv2.resize(img[:500], (img_size, img_size))
    return img

def create_train_data(img_size, color_type):
    start_time = time.time()
    # Loop over the training folder
    for class_ in range(n_classes):

        print('Loading directory c{}'.format(class_))

        files = glob(os.path.join('/usr/local/airflow/data/imgs/train', 'c' + str(class_), '*.jpg'))

        for file in files:
            img = get_cv2_image(file, img_size, color_type)
            train_images.append(img)
            train_labels.append(class_)

    print("Data Loaded in {} Min".format((time.time() - start_time) / 60))
    return train_images, train_labels


X, y = create_train_data(img_size, channel)
y = to_categorical(y, n_classes)

base_model  = tf.keras.applications.resnet.ResNet50(include_top = False,
                                                  weights = 'imagenet',
                                                  input_shape = (224,224,3))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)

output =tf.keras.layers.Dense(10,activation = tf.nn.softmax)(x)
model = tf.keras.models.Model(inputs=base_model.inputs, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False),
              metrics=['accuracy'])

model.summary()

# splitting train data to train and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15,shuffle=True)
print(X_train[0])
# convert data to numpy array for training
X_train = np.array(X_train, dtype=np.uint8).reshape(-1, img_size, img_size, channel)
X_valid = np.array(X_valid, dtype=np.uint8).reshape(-1, img_size, img_size, channel)

num_epochs = 5
def lr_schedule(epoch,lr):
    # Learning Rate Schedule

    lr = lr
    total_epochs = num_epochs

    check_1 = int(total_epochs * 0.9)
    check_2 = int(total_epochs * 0.8)
    check_3 = int(total_epochs * 0.6)
    check_4 = int(total_epochs * 0.4)

    if epoch > check_1:
        lr *= 1e-4
    elif epoch > check_2:
        lr *= 1e-3
    elif epoch > check_3:
        lr *= 1e-2
    elif epoch > check_4:
        lr *= 1e-1

    print("[+] Current Lr rate : {} ".format(lr))
    return lr
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

history = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_valid, y_valid),
    steps_per_epoch=16,
    batch_size=8,
    epochs=num_epochs,

    callbacks=[tensorboard],
    verbose=1)

