from src import util
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
import src.rnn_predict as predict
import keras.regularizers as reg
import numpy as np
import keras.backend as K

BATCH_SIZE = 256

difference_mat = np.zeros((BATCH_SIZE * 2 - 1, BATCH_SIZE * 2))
for i in range(difference_mat.shape[0]):
    for j in range(difference_mat.shape[1]):
        if i == j:
            difference_mat[i][j] = -1
        if i + 1 == j:
            difference_mat[i][j] = 1
print(difference_mat)

def custom_loss(y_true, y_pred):
    reg_weight = 0.01
    square_mat = K.dot(y_pred, K.transpose(y_pred))
    y_flat = y_pred[:,0:1]
    difference_mat_var = K.variable(difference_mat)
    difference_mat_var = tf.slice(difference_mat_var, [0, 0], K.shape(square_mat))
    difference_mat_var = difference_mat_var[:-1,:]
    return K.mean(K.square(y_pred - y_true), axis=-1) + reg_weight * K.mean(K.square(tf.matmul(difference_mat_var, y_flat)))

def computeActualFromOffset(originalMusStarts, originalMusLens, predictions):
    computed = []
    print(originalMusStarts)
    print(originalMusLens)
    for i in range(len(predictions)):
        tup = predictions[i]
        origStart = originalMusStarts[i]
        origLen = originalMusLens[i]
        computed.append([tup[0] + origStart, tup[1] + origLen])
    return computed

def main():
    util.weight_experiment(n_files=37)
    x_train, y_train = util.load_data_rnn(n_files=444)
    n_step_features = 4
    reg_weight = 0.01
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True, activation='relu', kernel_regularizer=reg.l2(0.001)),
                            input_shape=(util.TIMESTEPS * 2 + 1, n_step_features)))
    model.add(Bidirectional(LSTM(32, activation='relu', kernel_regularizer=reg.l2(0.01))))
    model.add(Dense(2, activation='linear', kernel_regularizer=reg.l2(0.05)))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    N_EPOCHS = 5
    history = model.fit(x_train, y_train, validation_split=0.1,
                        batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[], shuffle=True)

    model.save('rnn_model.h5')

    predict.predict(model)



# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()