from src.util import rnn_util as util, note_util as note
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
import keras.regularizers as reg
import numpy as np
from keras.initializers import glorot_uniform
from keras import optimizers
import keras.backend as K
from keras.models import load_model
import os, random, time, datetime
from threading import Thread
BATCH_SIZE = 128

def subset(x, y):
    return x[y[:, 2] != 1], y[y[:, 2] != 1]

def custom_loss(y_true, y_pred):
    penalty_weight = 0.01
    error = K.mean(K.square(y_pred[:, :2] - y_true[:, :2]), axis=-1)
    penalty = K.mean(K.square(y_true[:, 2] * (y_pred[:, :2] - y_true[:, :2])), axis=-1)
    return error# + penalty_weight * penalty

def train(load=False, bidirectional=False):
    n_step_features = 6
    reg_weight = 0.001
    if load:
        if not bidirectional:
            model = load_model("rnn_model.h5")
        else:
            model = load_model("birnn_model.h5")
    else:
        model = Sequential()
        init = glorot_uniform(seed=time.time())
        if not bidirectional:
            model.add(LSTM(32, return_sequences=True, activation='relu', kernel_regularizer=reg.l2(reg_weight),
                       kernel_initializer=init, input_shape=(util.TIMESTEPS + 1, n_step_features)))
            model.add(LSTM(32, return_sequences=False, activation='relu', kernel_regularizer=reg.l2(reg_weight),
                       kernel_initializer=init))
        else:
            model.add(
                Bidirectional(LSTM(32, return_sequences=True, activation='relu', kernel_regularizer=reg.l2(reg_weight),
                       kernel_initializer=init), input_shape=(util.TIMESTEPS * 2 + 1, n_step_features)))
            model.add(
                Bidirectional(LSTM(32, return_sequences=False, activation='relu', kernel_regularizer=reg.l2(reg_weight),
                               kernel_initializer=init)))

        model.add(Dense(3, activation='linear', kernel_regularizer=reg.l2(reg_weight), kernel_initializer=init))

        adam = optimizers.Adam(clipnorm=0.25)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=[])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    N_EPOCHS = 100
    get_subset = True
    n_files = len(os.listdir(os.getcwd() + '/mus'))
    sample_size = 200
    for_predict = False
    #load initial data
    x, y = util.load_data(files=random.sample(range(n_files), sample_size), for_predict=for_predict, bidirectional=bidirectional)
    start = datetime.datetime.now()
    for t in range(N_EPOCHS):
        print("Iteration {}".format(t))
        if get_subset:
            x, y = subset(x, y)
        if (datetime.datetime.now() - start) > datetime.timedelta(hours=8):
            exit(8)
        thread = Thread(target=util.load_data, args=(random.sample(range(n_files), sample_size), for_predict, bidirectional))
        thread.start()
        model.fit(x, y, validation_split=0.1,
                        batch_size=BATCH_SIZE, epochs=2, callbacks=[], shuffle=True)
        if not bidirectional:
            model.save('rnn_model.h5')
        else:
            model.save('birnn_model.h5')
        thread.join()
        assert(not np.array_equal(x, util.x_global))
        assert (not np.array_equal(y, util.y_global))
        x, y = util.x_global.copy(), util.y_global.copy()

    return model

def predict(model):
    musList, musNames, recList, recNames = util.loadSongLists(for_predict=True)
    for i in range(len(musList)):
        x_test, y_test = util.collectData([musList[i]], [recList[i]], bidirectional=True)
        predictions = np.zeros((x_test.shape[0] + util.TIMESTEPS, 3))
        predictions[:util.TIMESTEPS, :2] = x_test[0, :util.TIMESTEPS, 4:]
        for j in range(len(x_test)):
            # second half must be all zeros
            input = np.zeros((1, x_test.shape[1], x_test.shape[2]))
            input[0] = x_test[i]
            input[0, :util.TIMESTEPS + 1, 4:] = predictions[j: j + util.TIMESTEPS + 1, :2]
            prediction = model.predict_on_batch(input)
            predictions[j + util.TIMESTEPS] = prediction

        predictions[util.TIMESTEPS:] = model.predict(x_test)
        for j in range(len(y_test)):
            if y_test[j, 2] == 1:
                predictions[j + util.TIMESTEPS, 0] = predictions[j + util.TIMESTEPS - 1, 0]

        print('Mean loss: {}'.format(np.mean(np.square(y_test - predictions[util.TIMESTEPS:]))))
        # predictions = note.denormalizeTimes(predictions, recList[0][-1]['end'])
        # y_test = note.denormalizeTimes(y_test, recList[0][-1]['end'])
        print()
        print(predictions[20:25])
        print(y_test[:5])

        path = 'C://Users//cpgaf//OneDrive//Documents//NetBeansProjects//Expressive//files'
        for j in range(predictions.shape[0]):
            last_rec = recList[i][-1]['end']
            last_mus = musList[i][-1]['end']
            length = musList[i][j]['end'] - musList[i][j]['start']
            predicted_start = predictions[j][0] * last_rec / 100.0 + musList[i][j]['start'] * last_rec / last_mus
            musList[i][j]['start'] = predicted_start
            '''note.denormalizeTimeFromOffsetSubtract(predictions[j][0], musList[i][j]['start'],
                                                                            last_mus, last_rec)'''
            predicted_length = predictions[j][1] * last_rec / 100.0 + length * last_rec / last_mus
            '''musList[i][j]['end'] = musList[i][j]['start'] + note.denormalizeTimeFromOffsetSubtract(
                predictions[j][1], length,
                last_mus, last_rec)'''
            musList[i][j]['end'] = musList[i][j]['start'] + predicted_length
            if musList[i][j]['end'] - musList[i][j]['start'] <= 0:
                musList[i][j]['end'] = musList[i][j]['start'] + 100
        note.writePIDI(musList[i], path + "//predictions" + str(i) + ".txt")



def main():
    training_phase = False
    bidirectional = True
    if training_phase:
        train(load=True, bidirectional=bidirectional)
    else:
        if not bidirectional:
            model = load_model("rnn_model.h5")
        else:
            model = load_model("birnn_model.h5")
        predict(model)

# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()