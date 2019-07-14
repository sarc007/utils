import cv2
import tensorflow as tf
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
import tensorflow.keras
import time


# minst = tf.keras.datasets.mnist
# data = minst.load_data()
# check_data = data[1][0][0]
# print(check_data)
# print(len(check_data))
# check_data_np = np.asarray(data)
# print(check_data_np.shape)
# print(check_data_np[0])
from tensorflow.python.keras import layers

src_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\img_resized"
dst_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\data"
data_file = dst_dir + "\\" + "alphanumeric.hdf5"

f = h5py.File(data_file, 'r')
class_arr = f['class'][:]
labels_arr = f['img_labels'][:]
image_arr = f['img_dataset'][:]
f.close()
x_train, x_test, y_train, y_test = train_test_split(image_arr, labels_arr, test_size=0.2, random_state=42)
print(chr(y_train[200000]))
cv2.imshow('x train 0 ', x_train[200000])
cv2.waitKey(0)
BUFFER_SIZE = len(x_train)
BATCH_SIZE = 1536
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5 # Normalize the images to [-1, 1]
x_test = (x_test - 127.5) / 127.5 # Normalize the images to [-1, 1]
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.01),
  tf.keras.layers.Dense(36, activation=tf.nn.softmax)
])
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3000,
                    batch_size=100,
                    validation_data=(x_test, y_test),
                    verbose=1)
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print("Loss = {0}' , Accuracy = {1} ", val_loss, val_acc)
model.save('alphanum_reader.model')
new_model = tf.keras.models.load_model('alphanum_reader.model')
predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))


#model.add()

