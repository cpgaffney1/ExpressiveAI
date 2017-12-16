import util
import sys
from keras.models import Sequential
from keras import optimizers as opt
from keras.layers import LSTM, Dense
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import h5py
from keras.callbacks import TensorBoard


def mse_exclude_zeros(y_true, y_pred):
    indices = K.tf.where(K.tf.not_equal(y_true, 0))
    true = K.tf.gather_nd(y_true, indices)
    pred = K.tf.gather_nd(y_pred, indices)
    return K.mean(K.square(pred - true), axis=-1)

def main():
    musList, recList, matchesMapList, songNames = util.parseMatchedInput('javaOutput.txt')
    musList, recList = util.normalizeTimes(musList, recList)
    x, y, songLengths = util.dataAsSequential(musList, recList, matchesMapList)
    print(songLengths)

    x_train = x
    y_train = y
    model = Sequential()
    model.add(LSTM(util.H_LAYER_SIZE, return_sequences=True, input_shape=(util.TIMESTEPS, util.KEYBOARD * 3 + 1)))
    model.add(LSTM(util.H_LAYER_SIZE, activation='relu'))
    model.add(Dense(util.KEYBOARD, activation='linear'))
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model.compile(loss=mse_exclude_zeros, optimizer='adam', metrics=[])
    tensorboard = TensorBoard(log_dir='./Graph/rnn', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=util.BATCH_SIZE, epochs=util.N_EPOCHS,
              callbacks=[tensorboard, early_stopping])

    '''print(x_train.shape)
    worseCount = 0
    prevLoss = sys.maxsize
    for epoch in range(20):
        index = 0
        batch = 0
        print('EPOCH: ' + str(epoch))
        print('---------------------------')
        for length in songLengths:
            loss = model.train_on_batch(x_train[index : index + length], y_train[index : index + length])
            print('Batch: {}/{} ---> Loss: {}'.format(batch + 1, len(songLengths), loss))
            if loss > prevLoss:
                worseCount += 1
            else:
                worseCount = 0
            prevLoss = loss
            index += length
            batch += 1
        if worseCount >= 2 * len(songLengths):
            break
        print('')'''

    model.save('rnn_model.h5')

    '''musList, recList, matchesMapList = util.parseMatchedInput('testData.txt')
    assert(len(musList) == 1)
    assert(len(recList) == 1)
    musList, recList = util.normalizeTimes(musList, recList)
    ########### THIS IS BROKEN DO NOT USE ################
    x_test, y_test, chord_note_indices = util.dataAsSequentialPredict(musList[0], recList[0], matchesMapList)

    predictions = model.predict(x_test, batch_size=len(x_test))
    predictions = util.denormalizeTimes(predictions, musList[0][-1]['end'])
    print(K.get_value(K.print_tensor(predictions)))

    file = "C:\\Users\\cpgaffney1\\Documents\\NetBeansProjects\\ProjectMusic\\files\\predictions.txt"
    with open(file, 'w') as of:
        for i in range(len(chord_note_indices)):
            for j in range(len(chord_note_indices[i])):
                if chord_note_indices[i][j] != -1:
                    of.write("{},{}\n".format(chord_note_indices[i][j], predictions[i][j]))'''



# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()