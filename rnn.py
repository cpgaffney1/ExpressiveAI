import util
from keras.models import Model
from keras.layers import LSTM, Dense, Input
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import h5py
import keras.layers as layers
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


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
    mus_x_train, rec_x_train, core_train_features, y_train = util.load_data()

    mus_input = Input(shape=(2 * util.TIMESTEPS + 1, 4))
    rec_input = Input(shape=(util.TIMESTEPS + 1, 2))

    mus_lstm = Bidirectional(LSTM(4, return_sequences=True, activation='relu'),
                             input_shape=(util.TIMESTEPS * 2 + 1, 4))(mus_input)
    rec_lstm = LSTM(2, return_sequences=True, activation='relu', input_shape=(util.TIMESTEPS + 1, 2))(rec_input)

    concat = layers.concatenate([layers.Flatten()(mus_lstm), layers.Flatten()(rec_lstm)], axis=1)
    output = Dense(2, activation='linear')(concat)

    model = Model(inputs=[mus_input, rec_input], outputs=[output])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    N_EPOCHS = 1000
    history = model.fit([mus_x_train, rec_x_train], [y_train], validation_split=0.1,
                        batch_size=util.BATCH_SIZE, epochs=N_EPOCHS, callbacks=[early_stopping], shuffle=True)

    model.save('rnn_model.h5')

    #predict

    musList, recList, matchesMapList, songNames = util.parseMatchedInput('testData', [0])
    musList, recList = util.normalizeTimes(musList, recList)
    recList, matchesMapList = util.trim(recList, matchesMapList)
    musList, recList = util.addOffsets(musList, recList, matchesMapList)
    x, y = util.dataAsWindow(musList, recList, matchesMapList)
    x_test = x.astype('float32')
    mus_x_test, rec_x_test, core_test_features = util.splitData(x_test)
    mus_x_test, rec_x_test, core_test_features = mus_x_test.astype('float32'), rec_x_test.astype(
        'float32'), core_test_features.astype('float32')

    predictions = model.predict([mus_x_test, rec_x_test])

    actual = []
    mus = []
    for musNote in musList[0]:
        musIndex = musNote['index']
        if musIndex in matchesMapList[0].keys():
            recNote = recList[0][matchesMapList[0][musIndex]]
            actual.append(recNote['start_normal'])
            mus.append(musNote['start_normal'])
    start_pred = []
    for i in range(len(predictions)):
        start_pred.append(predictions[i][0])

    plt.plot(actual, label='actual')
    plt.plot(mus, label='input')
    plt.plot(start_pred, label='predictions')
    print()
    actual = actual[:len(actual) - (len(actual) - len(start_pred))]
    sq = [(actual[i] - start_pred[i]) ** 2 for i in range(len(actual))]
    print(np.mean(np.asarray(sq)))
    plt.legend()
    plt.show()


    file = "C://Users//cpgaffney1//Documents//NetBeansProjects//ProjectMusic//files//predictions.txt"
    with open(file, 'w') as of:
        for i in range(len(predictions)):
            of.write("{},{},{}\n".format(i, predictions[i][0], predictions[i][1]))



# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()