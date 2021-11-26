import pandas as pd
import numpy as np
import tensorflow as tf
import time
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from keras.models import Model
num_years = 11
num_days = 365
num_points = 96
raw_data = [None]*num_years
year_list = []
for i in range(num_years):
    year_list.append('D:\\2019 summer\\TensorFlow\\')
for i in range(2008, 2019, 1):
    year_list[i-2008] += str(i)
    year_list[i-2008] += '.xls'
for i in range(num_years):
    raw_data[i] = pd.read_excel(year_list[i])
process_data = [None]*11
for i in range(num_years):
    process_data[i] = np.array(raw_data[i])
usable_data = np.zeros((11, 365, 96))
zeros = []
for i in range(num_years):
    for j in range(num_days):
        for k in range(3, 99):
            if process_data[i][j][k] != 0:
                usable_data[i][j][k-3] = process_data[i][j][k]
            else:
                usable_data[i][j][k-3] = None
                zeros.append((i, j, k-3))
bucket = np.zeros(num_years)
for i in range(len(zeros)):
    bucket[zeros[i][0]] = 1
for i in range(num_years):
    if bucket[i] == 1:
        df_predict = pd.DataFrame(usable_data[i])
        predict = pd.DataFrame.interpolate(df_predict, method = 'spline', order = '5', axis = 0)
        for j in range(num_days):
            for k in range(num_points):
                usable_data[i][j][k] = predict[k][j]
#Leap years are validation sets.
#Five day is a group: use first 4 days to predict the 5th day.
#Suppliments are month, day of month, and day of week.

train_data = np.zeros((584, 384))
supp_train = np.zeros((584, 3))
train_answers = np.zeros((584, 96))
test_data = np.zeros((219, 384))
supp_test = np.zeros((219, 3))
test_answers = np.zeros((219, 96))
blank = np.zeros((11, 365))
x = 0
y = 0
for i in range(num_years):
    for j in range(num_days):
        if i % 4 == 0:
            if blank[i][j] == 1:
                continue
            else:
                test_data[x][0:96] = usable_data[i][j]
                test_data[x][96:192] = usable_data[i][j+1]
                test_data[x][192:288] = usable_data[i][j+2]
                test_data[x][288:384] = usable_data[i][j+3]
                test_answers[x] = usable_data[i][j+4]
                supp_test[x][0] = process_data[i][j][0].month
                supp_test[x][1] = process_data[i][j][0].day
                supp_test[x][2] = process_data[i][j][0].dayofweek
                blank[i][j:j+5] = 1
                x += 1
        else:
            if blank[i][j] == 1:
                continue
            else:
                train_data[y][0:96] = usable_data[i][j]
                train_data[y][96:192] = usable_data[i][j+1]
                train_data[y][192:288] = usable_data[i][j+2]
                train_data[y][288:384] = usable_data[i][j+3]
                train_answers[y] = usable_data[i][j+4]
                supp_train[y][0] = process_data[i][j][0].month
                supp_train[y][1] = process_data[i][j][0].day
                supp_train[y][2] = process_data[i][j][0].dayofweek
                blank[i][j:j+5] = 1
                y += 1

input1 = Input(shape=(384,), name = "input1")
input2 = Input(shape=(3,), name = "input2")
conv_out = Conv1D(filters = 10, kernel_size = 2, activation = 'relu', padding = 'same')(input1)
fc_in = keras.layers.concatenate([conv_out, input2])
output = Dense(200, activation = 'relu', name = "output")(fc_in)
model = Model(inputs = [input1, input2], outputs = [output])
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
model.fit([train_data, supp_train], [train_answers], batch_size = 73, epochs = 10, validation_data = ([test_data, supp_test], [test_answers]))