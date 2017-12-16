import util
import numpy as np
from keras import backend as K
from keras.models import h5py
from keras.models import load_model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def predict(seq_model, chord_model=None, core_input_shape=5):
    musList, recList, matchesMapList, songNames = util.parseMatchedInput('testData', [0])
    matchesMapList = [{i:i for i in range(len(musList[0]))}]
    musList, recList = util.normalizeTimes(musList, recList)
    recList, matchesMapList = util.trim(recList, matchesMapList)
    x, y = util.dataAsWindow(musList, recList, matchesMapList)
    x_test = x.astype('float32')
    mus_x_test, rec_x_test, core_test_features = util.splitData(x_test)
    mus_x_test, rec_x_test, core_test_features = mus_x_test.astype('float32'), rec_x_test.astype(
        'float32'), core_test_features.astype('float32')

    def to3d(arr):
        new = np.zeros((1, arr.shape[0], arr.shape[1]))
        new[0] = arr
        return new

    def to2d(arr):
        new = np.zeros((1, arr.shape[0]))
        new[0] = arr
        return new

    predictions = []
    for i in range(util.TIMESTEPS + 1):
        predictions.append(rec_x_test[1][i])

    for i in range(len(rec_x_test)):
        prev_predictions = to3d(rec_x_test[i])
        for j in range(util.TIMESTEPS + 1):
            prev_predictions[0][j] = predictions[-(util.TIMESTEPS+1) + j]
        # 0,1,2,3 is mus start, prev mus start, prev rec start, prev rec length, in chord
        if core_input_shape >= 3:
            core_test_features[i][2] = predictions[-1][0]
        if core_input_shape >= 4:
            core_test_features[i][3] = predictions[-1][1]
        #print(to3d(mus_x_test[i]))
        #print(prev_predictions)
        if core_test_features[i][0] == core_test_features[i][1] and chord_model != None or \
            chord_model == None:
            pred = seq_model.predict_on_batch([to3d(mus_x_test[i]), prev_predictions, to2d((core_test_features[i]))])
        elif chord_model != None:
            pred = chord_model.predict_on_batch([to3d(mus_x_test[i]), prev_predictions, to2d((core_test_features[i]))])
        predictions.append(pred[0])
        #print(pred[0])

    actual = []
    for musNote in musList[0]:
        musIndex = musNote['index']
        if musIndex in matchesMapList[0].keys():
            recNote = recList[0][matchesMapList[0][musIndex]]
            actual.append(recNote['start'])

    lastTime = recList[0][-1]['start']
    predictions = util.denormalizeTimes(predictions, lastTime)

    start_pred = []
    for i in range(len(predictions)):
        start_pred.append(predictions[i][0])

    plt.plot(actual, label='actual')
    plt.plot(start_pred, label='predictions')
    print()
    actual = actual[:len(actual) - (len(actual) - len(start_pred))]
    sq = [(actual[i]- start_pred[i]) ** 2 for i in range(len(actual))]
    print(np.mean(np.asarray(sq)))
    plt.legend()
    plt.savefig("prediction_vs_actual_plot")

    #print('lengths: rec_x_test = {}, predictions = {}, mus = {}'.format(len(rec_x_test), len(predictions), len(musList[0])))
    file = "C://Users//cpgaffney1//Documents//NetBeansProjects//ProjectMusic//files//predictions.txt"
    with open(file, 'w') as of:
        for i in range(len(predictions) - 1):
            of.write("{},{},{}\n".format(i, predictions[i][0], predictions[i][1]))



def main():
    ################
    oneModel = True  ##
    ################
    if oneModel:
        model = load_model('good_models/predicts_chords_model.h5')
        ###
        '''
        musList, recList, matchesMapList, songNames = util.parseMatchedInput('javaOutput', [4,10,17])
        musList, recList = util.normalizeTimes(musList, recList)
        recList, matchesMapList = util.trim(recList, matchesMapList)
        x, y = util.dataAsWindow(musList, recList, matchesMapList)
        x_test = x.astype('float32')
        mus_x_test, rec_x_test, core_test_features = util.splitData(x_test)
        mus_x_test, rec_x_test, core_test_features = mus_x_test.astype('float32'), rec_x_test.astype(
            'float32'), core_test_features.astype('float32')
        print(model.evaluate(x=[mus_x_test, rec_x_test, core_test_features], y=y))
        '''
        ###

        print(model.get_layer("output").get_weights())
        predict(model)
    else:
        seq_model = load_model('enhanced_nn_model_seq.h5')
        chord_model = load_model('enhanced_nn_model_chord.h5')
        predict(seq_model, chord_model=chord_model)


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()