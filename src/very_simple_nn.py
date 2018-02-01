from src import util
from keras.models import Sequential
from keras.layers import Dense
import nn_predict as predict

BATCH_SIZE = 512

def main():
    musList, recList, matchesMapList = util.parseMatchedInput('javaOutput.txt')
    musList, recList = util.normalizeTimes(musList, recList)
    x, y = util.dataAsWindow(musList, recList, matchesMapList, verySimple=True)

    x_train = x
    y_train = y

    model = Sequential()

    input_dim = util.V_SIMPLE_FEATURES
    model.add(Dense(1, activation='linear', input_dim=input_dim))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    N_EPOCHS = 50
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
              callbacks=[], shuffle=True)

    model.save('very_simple_nn_model.h5')


    predict.predict(model, verySimple=True)


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()