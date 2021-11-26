import h5py
filename = 'my_model_weights.h5'
h5 = h5py.File(filename, 'r')
x = h5[0]