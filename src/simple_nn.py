from src import util
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.initializers import TruncatedNormal
import nn_predict as predict

BATCH_SIZE = 512

def main():
    simplified = False

    musList, recList, matchesMapList = util.parseMatchedInput('javaOutput.txt')
    musList, recList = util.normalizeTimes(musList, recList)
    x, y = util.dataAsWindow(musList, recList, matchesMapList, simplified=simplified)

    x_train = x
    y_train = y
    print(x_train[:5])
    print(y_train[:5])

    model = Sequential()

    init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    input_dim = util.NON_SPARSE_FEATURES if not simplified else util.SIMPLIFIED_FEATURES
    output = util.NON_SPARSE_FEATURES * 10
    model.add(Dense(output, activation='relu', input_dim=input_dim, kernel_initializer=init))
    model.add(Dropout(0.4))
    '''model.add(Dense(output, activation='relu', kernel_initializer=init))
    if not simplified:
        output = 2 * util.TIMESTEPS + 1
        model.add(Dense(output, activation='relu', kernel_initializer=init))
    model.add(Dropout(0.5))'''
    model.add(Dense(1, activation='linear'))

    early_stopping = EarlyStopping(monitor='loss', patience=10)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    N_EPOCHS = 30
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
              callbacks=[], shuffle=True)

    if not simplified:
        model.save('simple_nn_model.h5')
    else:
        model.save('simple_nn_model_simple.h5')

    predict.predict(model, simplified=simplified)


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()