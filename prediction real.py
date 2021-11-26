import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
import json, codecs
file_path = ["traindata.json", "trainanswer.json","testdata.json", "testanswer.json"]
obj_text = []
for i in range(4):
    obj_text.append(codecs.open(file_path[i], 'r', encoding='utf-8').read())
train_data = np.array(json.loads(obj_text[0]))
train_answers = np.array(json.loads(obj_text[1]))
test_data = np.array(json.loads(obj_text[2]))
test_answers = np.array(json.loads(obj_text[3]))

model = Sequential()
model.add(Dense(32, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
model.add(Dense(1, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
model.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam(), metrics = ['mape'])
model.fit(train_data, train_answers, batch_size = 50, epochs = 20, validation_data = (test_data, test_answers))