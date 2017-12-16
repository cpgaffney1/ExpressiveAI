import util
import random
import numpy as np
from keras import backend as K
from keras.models import h5py
from keras.models import load_model
import tensorflow as tf
from keras.utils import plot_model
import keras

def mse_exclude_zeros(y_true, y_pred):
    indices = K.tf.where(K.tf.not_equal(y_true, 0))
    true = K.tf.gather_nd(y_true, indices)
    pred = K.tf.gather_nd(y_pred, indices)
    return K.mean(K.square(pred - true), axis=-1)

model = load_model('rnn_model.h5', custom_objects={'mse_exclude_zeros':mse_exclude_zeros})

keras.callbacks.TensorBoard(log_dir='/Graph', histogram_freq=0,
          write_graph=True, write_images=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,
          write_graph=True, write_images=True)
tbCallBack.set_model(model)


musList, recList, matchesMapList = util.parseMatchedInput('testData.txt')
assert(len(musList) == 1)
assert(len(recList) == 1)
musList, recList = util.normalizeTimes(musList, recList)
x_test, y_test, chord_note_indices = util.transformInput(musList, recList, matchesMapList)


#y_test = K.constant(y_test)

#print(K.get_value(mse(y_pred, y_true)))
predictions = model.predict(x_test, batch_size=1)
'''print(x_test.shape)
print(y_test.shape)
print(predictions.shape)
print(K.get_value(y_test))'''

#print(K.get_value(mse_exclude_zeros(y_test, predictions)))
predictions = util.denormalizeTimes(predictions, musList[0][-1]['end'])

file = "C:\\Users\\cpgaffney1\\Documents\\NetBeansProjects\\ProjectMusic\\files\\predictions.txt"
with open(file, 'w') as of:
    for i in range(len(chord_note_indices)):
        for j in range(len(chord_note_indices[i])):
            if chord_note_indices[i][j] != -1:
                of.write("{},{}\n".format(chord_note_indices[i][j], predictions[i][j]))