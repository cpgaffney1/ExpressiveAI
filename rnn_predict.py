import util
from keras.models import Model
from keras.layers import LSTM, Dense, Input
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


def predict():
    # predict

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

    print(predictions)

    file = "C://Users//cpgaffney1//Documents//NetBeansProjects//ProjectMusic//files//predictions.txt"
    with open(file, 'w') as of:
        for i in range(len(predictions)):
            of.write("{},{},{}\n".format(i, predictions[i][0], predictions[i][1]))

model = load_model("rnn_model.h5")
