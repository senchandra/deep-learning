# import libraries
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# import keras libraries
from keras.preprocessing.image import load_img
from tensorflow.keras import layers, models, datasets

# define image path
images_path = 'images/'

# create images containing all the image
num_images = len(os.listdir(images_path))
images = np.empty(shape=(180,72,72,3), dtype=int)
for i in range(len(os.listdir(images_path))):
    images[i] = np.array(load_img('{}{}.jpg'.format(images_path, i)))

# view an image
import matplotlib.pyplot as plt
plt.imshow(images[10])

# create labels for the image, the images are all incremented by 1 degree with name as 1.jpg
labels = np.linspace(0, 179, 180)

# reshape X and Y
X = images.reshape(180, 72, 72, 3)
Y = labels.reshape(180, 1)

# import keras layers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras import backend as K 
from tensorflow.keras.callbacks import EarlyStopping

# clear Keras session and define a sequential model
K.clear_session()
model = Sequential()

# add Conv and Dense layers to the module
model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(72,72,3)))
model.add(MaxPool2D(pool_size=(2,2), padding="valid"))
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), padding="valid"))
model.add(Conv2D(128, kernel_size=3, activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), padding="valid"))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))
# Remove activation for the below two layers to make it a Regression problem instead of classification
model.add(Dense(180))
model.add(Dense(1))

# compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

# print the layers in the model
model.summary()

# prepare train, dev and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
X_dev_set, X_test_set, Y_dev_set, Y_test_set = train_test_split(X_test, Y_test, test_size = 0.5, random_state=100)

# define earlystopping for the model
es = EarlyStopping(monitor='loss', mode='min', patience=20, min_delta=1)

# train the model on train set cross-validated on dev set
model.fit(X_train, Y_train, validation_data=(X_dev_set, Y_dev_set), epochs=100, batch_size=16, callbacks=[es])

# plot loss vs epoch
plt.plot(model.history.history['loss'])

# predict the labels 
print(model.predict(X_test_set))

# view the original labels
print(Y_test_set)