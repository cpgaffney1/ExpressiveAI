from src import rnn_util as util
from src import note_util as note
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Lambda
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
import keras.regularizers as reg
import numpy as np
import keras.backend as K
from keras.models import load_model
import os, random, time, datetime
from threading import Thread
BATCH_SIZE = 256


def train():
    n_step_features = 6
    reg_weight = 0.00
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True, activation='relu', kernel_regularizer=reg.l2(reg_weight)),
                            input_shape=(util.TIMESTEPS * 2 + 1, n_step_features)))
    model.add(Dropout(0.6))
    model.add(Bidirectional(LSTM(32, return_sequences=False, activation='relu', kernel_regularizer=reg.l2(reg_weight))))
    model.add(Dropout(0.6))
    model.add(Dense(2, activation='linear', kernel_regularizer=reg.l2(reg_weight)))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    N_EPOCHS = 50
    n_files = len(os.listdir(os.getcwd() + '/mus'))
    sample_size = 200
    #load initial data
    x, y = util.load_data(files=random.sample(range(n_files), sample_size), for_predict=False)
    start = datetime.datetime.now()
    for t in range(N_EPOCHS):
        if (datetime.datetime.now() - start) > datetime.timedelta(hours=8):
            exit(8)
        thread = Thread(target=util.load_data, args=(random.sample(range(n_files), sample_size), False))
        thread.start()
        model.fit(x, y, validation_split=0.1,
                        batch_size=BATCH_SIZE, epochs=1, callbacks=[], shuffle=True)
        model.save('rnn_model.h5')
        thread.join()
        x, y = util.x_global.copy(), util.y_global.copy()

    return model

# uses only mus information to predict
def predict(model):
    musList, musNames, recList, recNames = util.loadSongLists(for_predict=True)
    x_test, y_test = util.collectData(musList, recList)
    predictions = np.zeros((x_test.shape[0] + util.TIMESTEPS, 2))
    predictions[:util.TIMESTEPS * 2 + 1] = x_test[0, :, 4:]
    for i in range(len(x_test)):
        #second half must be all zeros
        input = np.zeros((1, x_test.shape[1], x_test.shape[2]))
        input[0] = x_test[i]
        input[0, :util.TIMESTEPS + 1, 4:] = predictions[i : i + util.TIMESTEPS + 1, :]
        prediction = model.predict_on_batch(input)
        predictions[i + util.TIMESTEPS] = prediction

    predictions[util.TIMESTEPS:] = model.predict(x_test)
    print('Mean loss: {}'.format(np.mean(np.abs(y_test - predictions[util.TIMESTEPS:]))))
    print(predictions[:5])
    print(y_test[:5])
    predictions = note.denormalizeTimes(predictions, recList[0][-1]['end'])
    y_test = note.denormalizeTimes(y_test, recList[0][-1]['end'])
    print(musList[0][:5])
    print(recList[0][:5])
    print(x_test[0])
    print(predictions[:5])
    print(y_test[:5])
    predictions[util.TIMESTEPS:] = y_test
    path = 'C://Users//cpgaf//OneDrive//Documents//NetBeansProjects//Expressive//files'
    ######
    ### TODO undefined behavior on more than one song
    for i in range(len(musList)):
        for j in range(predictions.shape[0]):
            length = musList[i][j]['end'] - musList[i][j]['start']
            musList[i][j]['start'] += predictions[j][0]
            musList[i][j]['end'] = musList[i][j]['start'] + length + predictions[j][1]
            print(musList[i][j])
        note.writePIDI(musList[i], path + "//predictions" + str(i) + ".txt")


def main():
    #model = train()
    model = load_model("rnn_model.h5")
    predict(model)

# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()