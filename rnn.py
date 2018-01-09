import util
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import rnn_predict as predict


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
    x_train, y_train = util.load_data_rnn()
    exit()
    model = Sequential()
    model.add(Bidirectional(LSTM(8, return_sequences=True, activation='relu'),
                             input_shape=(util.TIMESTEPS * 2 + 1, 6)))
    model.add(Bidirectional(LSTM(4, return_sequences=True, activation='relu'),
                            input_shape=(util.TIMESTEPS * 2 + 1, 6)))
    model.add(Dense(2, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    N_EPOCHS = 1000
    history = model.fit([x_train], [y_train], validation_split=0.1,
                        batch_size=util.BATCH_SIZE, epochs=N_EPOCHS, callbacks=[early_stopping], shuffle=False)

    model.save('rnn_model.h5')

    predict.predict(model)



# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()