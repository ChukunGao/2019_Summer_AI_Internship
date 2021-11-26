import pandas as pd
import numpy as np
import tensorflow as tf
import time
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from keras.models import Model
import codecs, json
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
            if process_data[i][j][k] != 0 and str(process_data[i][j][k]) != 'nan':
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
#Suppliments are month, day of month, day of week, and time of day.
train_data = np.zeros((279552, 99))
train_answers = np.zeros((279552, 1))
test_data = np.zeros((104832, 99))
test_answers = np.zeros((104832, 1))
temp = np.zeros((96, 2))
x = 0
y = 0

for i in range(num_years):
    for j in range(num_days):
        for k in range(num_points):
            if i % 4 == 0:
                if j != 364:
                    for p in range(96-k):
                        temp[p] = [j, k+p]
                    if k != 0:
                        for p in range(k):
                            temp[p+96-k] = [j+1, p]
                    for p in range(95):
                        test_data[x][p] = usable_data[i][int(temp[p][0])][int(temp[p][1])]
                        test_data[x][95] = process_data[i][j][0].month
                        test_data[x][96] = process_data[i][j][0].day
                        test_data[x][97] = process_data[i][j][0].dayofweek
                        test_data[x][98] = k
                        test_answers[x][0] = usable_data[i][int(temp[95][0])][int(temp[95][1])]
                    x += 1
            else:
                if j != 364:
                    for p in range(96-k):
                        temp[p] = [j, k+p]
                    if k != 0:
                        for p in range(k):
                            temp[p+96-k] = [j+1, p]
                    for p in range(95):
                        train_data[y][p] = usable_data[i][int(temp[p][0])][int(temp[p][1])]
                        train_data[y][95] = process_data[i][j][0].month
                        train_data[y][96] = process_data[i][j][0].day
                        train_data[y][97] = process_data[i][j][0].dayofweek
                        train_data[y][98] = k
                        train_answers[y][0] = usable_data[i][int(temp[95][0])][int(temp[95][1])]
                    y += 1
train_data = train_data.tolist()
train_answers = train_answers.tolist()
test_data = test_data.tolist()
test_answers = test_answers.tolist()
file_path = ["traindata.json", "trainanswer.json","testdata.json", "testanswer.json"]
json.dump(train_data, codecs.open(file_path[0], 'w', encoding = 'utf-8'))
json.dump(train_answers, codecs.open(file_path[1], 'w', encoding = 'utf-8'))
json.dump(test_data, codecs.open(file_path[2], 'w', encoding = 'utf-8'))
json.dump(test_answers, codecs.open(file_path[3], 'w', encoding = 'utf-8'))