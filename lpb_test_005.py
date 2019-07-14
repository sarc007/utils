from tensorflow.keras.layers import (Conv2D, Flatten, Lambda, Dense, concatenate, Dropout, Input)
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import cv2
import os


def label_img(img):
	word_label = img.split('.')[-3]
	if word_label == 'r':
		return 1
	elif word_label == 'i':
		return 0


train_directory = '/train'
images = []
y = []

dataset = pd.read_csv('features.csv')

dataset = dataset[['first_value', 'second_value']]

features = dataset.iloc[:, 0:2].values

for root, dirs, files in os.walk(train_directory):
	for file in files:
		image = cv2.imread(root + '/' + file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
		images.append(image)
		label = label_img(file)
		y.append(label)

images = np.asarray(images)
images = images.reshape((-1, 512, 512, 1))

image_input = Input(shape=(512, 512, 1))
aux_input = Input(shape=(2,))

input_layer = Conv2D(32, (5, 5), activation='relu')(image_input)
cov1 = Conv2D(24, (5, 5), activation='relu', subsample=(2, 2))(input_layer)
cov2 = Conv2D(36, (5, 5), activation='relu', subsample=(2, 2))(cov1)
cov3 = Conv2D(48, (5, 5), activation='relu', subsample=(2, 2))(cov2)
cov4 = Conv2D(64, (5, 5), activation='relu')(cov3)
cov5 = Conv2D(64, (3, 3), activation='relu')(cov4)
dropout = Dropout(0.5)(cov5)
flatten = Flatten()(dropout)

# Here we add in the feature vectors
merge = concatenate([flatten, aux_input])

d1 = Dense(100, activation='elu')(merge)
d2 = Dense(50, activation='elu')(d1)
d3 = Dense(10, activation='elu')(d2)
out = Dense(1)(d3)

model = Model(inputs=[image_input, aux_input], outputs=[out])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit([images, features], y, epochs=50)
