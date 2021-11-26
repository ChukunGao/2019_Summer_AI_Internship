import numpy as np
x = np.zeros((10, 20, 30))
for i in range(10):
    for j in range(20):
        for k in range(30):
            x[i][j][k] = 1
            print(i, j, k, len(x[i][j]))