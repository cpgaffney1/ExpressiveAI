from src import util
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Flatten
from keras.initializers import TruncatedNormal
from keras.layers import Bidirectional
import src.rnn_predict as predict
import keras.regularizers as reg
import numpy as np
import time
from keras.utils import to_categorical
from keras.models import load_model
import keras.backend as K

train = True

def construct_dataset():
    n_files = 9
    for_predict = False
    is_ml = False

    musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList = util.parseMatchedInput(
        'C://Users//cpgaf//PycharmProjects//ExpressiveAI//javaOutput/javaOutput', range(0, n_files))
    musList, recList = util.normalizeTimes(musList, recList)
    musList, recList = util.normalizeIndices(musList, recList)

    x_raw_list = []
    x_list = []
    y_list = []
    obs_list = []
    for songIndex in range(len(matchesMapList)):
        x_raw_i, x_i, y_i, n_obs = util.assemble_matching_data_for(songIndex, musList, recList, potentialMatchesMapList, matchesMapList,
                                       for_predict, is_ml)
        x_raw_list.append(x_raw_i)
        x_list.append(x_i)
        y_list.append(y_i)
        obs_list.append(n_obs)

    x_raw = np.zeros((sum(obs_list), 2 * util.TIMESTEPS + 1, 2 * util.N_MATCHING_FEATURES))
    x = np.zeros((sum(obs_list), util.N_ML_FEATURES))
    y = np.zeros(sum(obs_list))

    start = 0
    for i in range(len(x_list)):
        end = start + obs_list[i]
        x_raw[start:end] = x_raw_list[i]
        x[start:end] = x_list[i]
        y[start:end] = y_list[i]
        start = end

    print(x_raw[0])
    print(y[0])
    print('\n')
    print(x_raw[1])
    print(y[1])
    print('\n')
    print(x_raw[2])
    print(y[2])
    print('\n')
    print(x_raw[3])
    print(y[3])
    print('\n')
    print(x_raw[4])
    print(y[4])
    print('\n')

    np.save('../data/x_train', x_raw)
    np.save('../data/x_train_ml', x)
    np.save('../data/y_train', y)

def load_data():
    x_raw = np.load('../data/x_train.npy')
    x_train = np.load('../data/x_train_ml.npy')
    y_train = np.load('../data/y_train.npy')
    shuffle_indices = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_indices)

    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    x_raw = x_raw[shuffle_indices]
    return x_train, y_train, x_raw


def train_nn():
    x_train, y_train, x_raw = load_data()

    y_train = to_categorical(y_train, num_classes=2)

    model = Sequential()
    np.random.seed(seed=int(time.time()))
    # init = TruncatedNormal(mean=0.0, stddev=0.1, seed=None)

    model.add(Dense(8, activation='relu', input_shape=(util.TIMESTEPS * 2 + 1, 2 * util.N_MATCHING_FEATURES)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    N_EPOCHS = 30
    BATCH_SIZE = 256

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(x_raw, y_train, validation_split=0.15, epochs=N_EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[early_stopping], shuffle=True)

    '''model.add(Dense(2, activation='softmax', input_shape=(util.N_ML_FEATURES,),
                    kernel_initializer=init, kernel_regularizer=reg.l2()))

    N_EPOCHS = 3
    BATCH_SIZE = 256

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit(x_train, y_train, validation_split=0.15, epochs=N_EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[], shuffle=True)'''

    model.save("matching_model.h5")

    y_pred = model.predict(x_raw)

    print(y_pred)
    print(np.count_nonzero(np.ndarray.astype(np.round(y_pred[:, 1]), np.int)))

def main():
    if train:
        train_nn()
    else:
        construct_dataset()

# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()