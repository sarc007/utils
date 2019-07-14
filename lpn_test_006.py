import cv2
import tensorflow as tf
import os
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
model_dir = "F:\\PycharmProjects\\ai_Lessons\\OpenCV_3_KNN_Character_Recognition_Python-master\\model"
data_file = dst_dir + "\\" + "alphanumeric.hdf5"

f = h5py.File(data_file, 'r')
class_arr = f['class'][:]
labels_arr = f['img_labels'][:]
image_arr = f['img_dataset'][:]
f.close()
x_train, x_test, y_train, y_test = train_test_split(image_arr, labels_arr, test_size=0.2, random_state=42)
# print(chr(y_train[200000]))
# cv2.imshow('x train 0 ', x_train[200000])
# cv2.waitKey(0)
x_train = (x_train - 127.5) / 127.5 # Normalize the images to [-1, 1]
x_test = (x_test - 127.5) / 127.5 # Normalize the images to [-1, 1]
# i = 0
# for y in y_train:
# 	if y < 58:
# 		y_train[i] = y_train[i]-48
# 	else :
# 		y_train[i] = y_train[i] - 55
# 	i += 1
# i = 0
# for y in y_test:
# 	if y < 58:
# 		y_test[i] = y_test[i]-48
# 	else :
# 		y_test[i] = y_test[i] - 55
# 	i += 1
y_train[y_train < 58] -= 48
y_train[y_train > 64] -= 55

y_test[y_test < 58] -= 48
y_test[y_test > 64] -= 55

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
  tf.keras.layers.Dropout(0.01),
  tf.keras.layers.Dense(36, activation=tf.nn.softmax)
])
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50,
                    validation_data=(x_test, y_test),
                    verbose=1)
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss = {0}' , Accuracy = {1} ", val_loss, val_acc)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(model_dir + '\\alphanum_reader.model')
new_model = tf.keras.models.load_model(model_dir + '\\alphanum_reader.model')
predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))
pred_ascii = predictions
pred_ascii[pred_ascii < 10] += 48
pred_ascii[pred_ascii > 9] += 55
print(chr(pred_ascii[0]))
cv2.imshow('x test 0 ', (x_test[0] * 127.5) + 127.5)
cv2.waitKey(0)


#model.add()

