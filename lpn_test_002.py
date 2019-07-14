import cv2
import tensorflow as tf
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# minst = tf.keras.datasets.mnist
# data = minst.load_data()
# check_data = data[1][0][0]
# print(check_data)
# print(len(check_data))
# check_data_np = np.asarray(data)
# print(check_data_np.shape)
# print(check_data_np[0])
src_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\img_resized"
dst_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\data"
data_file = dst_dir + "\\" + "alphanumeric.hdf5"

f = h5py.File(data_file, 'r')
class_arr = f['class'][:]
labels_arr = f['img_labels'][:]
image_arr = f['img_dataset'][:]
f.close()
x_train, x_test, y_train, y_test = train_test_split(image_arr, labels_arr, test_size=0.2, random_state=42)
print(chr(y_train[0]))
cv2.imshow('x train 0 ', x_train[0])
cv2.waitKey(0)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(784, activation=tf.nn.leaky_relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(36, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,  epochs=500)
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss = {0}' , Accuracy = {1} ", val_loss, val_acc)
model.save('alphanum_reader.model')
new_model = tf.keras.models.load_model('alphanum_reader.model')
predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))


#model.add()

