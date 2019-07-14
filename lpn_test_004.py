import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
src_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\img_resized"
dst_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\data"
data_file = dst_dir + "\\" + "alphanumeric-20190704.hdf5"

f = h5py.File(data_file, 'r')
class_arr = f['class'][:]
labels_arr = f['img_labels'][:]
image_arr = f['img_dataset'][:]
f.close()
print("Loaded HDF5 File")
x_train, x_test, y_train, y_test = train_test_split(image_arr, labels_arr, test_size=0.3, random_state=42)


# %matplotlib inline # Only use this if using iPython
image_index = 7777 # You may select anything up to 60,000
print(chr(y_train[image_index])) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, activation=tf.nn.sigmoid, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, activation=tf.nn.sigmoid, kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())# Flattening the 2D arrays for fully connected layers
model.add(Dense(512, activation=tf.nn.tanh))
model.add(Dropout(0.2))
model.add(Dense(36, activation=tf.nn.softmax))
model.compile(optimizer='adam', metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')
model.fit(x=x_train,y=y_train, epochs=36)
model.evaluate(x_test, y_test)

image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())