from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adadelta, SGD, Adam
from keras.models import model_from_json
from keras.datasets import fashion_mnist
from keras import backend as K

from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def stepFunction(array):
    for i in range(len(array)):
        array[i] = 1 if array[i]>0.5 else 0
    return array

def loadTraining():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    Y_train = tf.keras.utils.to_categorical(Y_train, 10)
    Y_test = tf.keras.utils.to_categorical(Y_test, 10)
    X_test = X_test / 255
    X_train = X_train / 255
    return (X_train, Y_train), (X_test, Y_test)

def save_model(model, network_path):
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    open(os.path.join(network_path, 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join(network_path, 'weights.h5'), overwrite=True)

def read_model(network_path):
    model = model_from_json(open(os.path.join(network_path, 'architecture.json')).read())
    model.load_weights(os.path.join(network_path, 'weights.h5'))
    return model


def readImageFromFile(filepath):
    img = cv2.imread(filepath, 0)
    img2 = []

    for i in range(len(img)):
        img2 += list(img[i])

    img = np.array(img2)
    img = img/255.0
    img = img.reshape(28, 28, 1)
    return img

'''
#Load training data
(X_train, Y_train), (X_test, Y_test) = loadTraining()

# Build model
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[28, 28, 1]))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# Train model and plot result
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=256, epochs=10, validation_data=(X_test, Y_test))

# Save model
save_model(model, "model")

'''

#Read model
model = read_model("model")
input = np.array([readImageFromFile("image/nikesb3.png")])
output = model.predict(input)[0]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(np.around(output, decimals=1))
print(stepFunction(output))

print("It is a:", class_names[np.argmax(output)] )

''
