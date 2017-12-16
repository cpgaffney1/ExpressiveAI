import util
import numpy as np
from keras import backend as K
from keras.models import h5py
from keras.models import load_model
import matplotlib.pyplot as plt

def predict(model, simplified=False, verySimple=False):
    musList, recList, matchesMapList = util.parseMatchedInput('testData.txt')
    assert (len(musList) == 1)
    assert (len(recList) == 1)
    musList, recList = util.normalizeTimes(musList, recList)
    x_test, initial_predictions, musNoteIndices = util.dataAsWindowForPredict(musList[0], recList[0], matchesMapList[0],
                                                                              simplified=simplified, verySimple=verySimple)
    print(x_test)
    predictions = []
    for pred in initial_predictions:
        predictions.append(pred[0])
    prev_predictions = initial_predictions
    for i in range(len(x_test)):
        correspondingMusIndex = musNoteIndices[i]
        if correspondingMusIndex not in matchesMapList[0].keys():
            continue
        index = 0
        for j in range(0, len(x_test[i]), 2):
            if x_test[i][j] == -1:
                x_test[i][j] = prev_predictions[index][0]
                x_test[i][j + 1] = prev_predictions[index][1]
                index += 1
        if i > 995 and i < 1005:
            print("prev predictions")
            print(prev_predictions)
            print("x test at i")
            print(x_test[i])
        pred = model.predict_on_batch(np.reshape(x_test[i], (1, len(x_test[i]))))
        predictions.append(pred[0][0])
        print("Prediction:" + str(pred))
        if i < 1000:
            pred = recList[0][matchesMapList[0][correspondingMusIndex]]['start_normal']
        else:
            pred = pred[0][0]
        prev_predictions = prev_predictions[1:]
        note_len = musList[0][correspondingMusIndex]['end_normal'] - musList[0][correspondingMusIndex]['start_normal']
        prev_predictions.append((pred, pred + note_len))

    actual = []
    for musNote in musList[0]:
        musIndex = musNote['index']
        if musIndex in matchesMapList[0].keys():
            recNote = recList[0][matchesMapList[0][musIndex]]
            actual.append(recNote['start_normal'])

    predictions = np.reshape(np.asarray(predictions), (len(predictions), 1))

    print(predictions)
    plt.plot(actual)
    plt.plot(predictions)
    print()
    print(np.mean(np.square(np.asarray(actual[:len(actual) - (len(actual) - len(predictions))]) - np.asarray(predictions))))
    print(K.get_value(K.mean(K.square(np.asarray(actual[:len(actual) - (len(actual) - len(predictions))]) - np.asarray(predictions)),axis=-1)))
    plt.show()

    predictions = util.denormalizeTimes(predictions, recList[0][-1]['end'])

    print('lengths: x_test = {}, predictions = {}, mus = {}'.format(len(x_test), len(predictions), len(musList[0])))
    file = "C:\\Users\\cpgaffney1\\Documents\\NetBeansProjects\\ProjectMusic\\files\\predictions.txt"
    with open(file, 'w') as of:
        for i in range(len(predictions)):
            of.write("{},{}\n".format(i, predictions[i][0]))


def main():
    simplified = False
    if simplified:
        model = load_model('conv_model_simple.h5')
    else:
        model = load_model('conv_model.h5')
    predict(simplified, model)


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()