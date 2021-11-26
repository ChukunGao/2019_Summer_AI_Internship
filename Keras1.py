import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten

#The pictures are 28*28

(TrainX, TrainY), (TestX, TestY) = mnist.load_data()
TrainX = TrainX.astype('float32')
TrainY = TrainY.astype('float32')
TrainX = TrainX/255.0
TestX = TestX/255.0
TrainY = keras.utils.to_categorical(TrainY, 10)
TestY = keras.utils.to_categorical(TestY, 10)

model = Sequential()
model.add(Flatten())
model.add(Dense(500, activation = 'tanh'))
model.add(Dense(500, activation = 'tanh'))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
model.fit(TrainX, TrainY, batch_size = 200, epochs = 50, validation_data = (TestX, TestY))
score = model.evaluate(TestX, TestY)
print(score)
