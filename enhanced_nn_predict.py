import util
import numpy as np
from keras import backend as K
from keras.models import h5py
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 512

difference_mat = np.zeros((BATCH_SIZE * 2 - 1, BATCH_SIZE * 2))
for i in range(difference_mat.shape[0]):
    for j in range(difference_mat.shape[1]):
        if i == j:
            difference_mat[i][j] = -1
        if i + 1 == j:
            difference_mat[i][j] = 1
print(difference_mat)

def custom_loss(y_true, y_pred):
    reg_weight = 0.01
    square_mat = K.dot(y_pred, K.transpose(y_pred))
    y_flat = y_pred[:,0:1]
    difference_mat_var = K.variable(difference_mat)
    difference_mat_var = tf.slice(difference_mat_var, [0, 0], K.shape(square_mat))
    difference_mat_var = difference_mat_var[:-1,:]
    return K.mean(K.square(y_pred - y_true), axis=-1) + reg_weight * K.mean(K.square(tf.matmul(difference_mat_var, y_flat)))

def computeActualFromOffset(originalMusStarts, originalMusLens, predictions):
    computed = []
    for i in range(len(predictions)):
        tup = predictions[i]
        origStart = originalMusStarts[i]
        origLen = originalMusLens[i]
        computed.append([tup[0] + origStart, tup[1] + origLen])
    return computed

def predict(seq_model, core_input_shape=5, withOffset=False):
    musList, recList, matchesMapList, songNames = util.parseMatchedInput('testData', [0, 1])
    #matchesMapList = [{i:i for i in range(len(musList[0]))}]
    musList, recList = util.normalizeTimes(musList, recList)
    recList, matchesMapList = util.trim(recList, matchesMapList)
    if withOffset:
        recList = util.addOffsets(musList, recList, matchesMapList)
    for i in range(len(musList)):
        x, y = util.dataAsWindow([musList[i]], [recList[i]], [matchesMapList[i]])
        x_test = x.astype('float32')
        mus_x_test, rec_x_test, core_test_features = util.splitData(x_test)
        mus_x_test, rec_x_test, core_test_features = mus_x_test.astype('float32'), rec_x_test.astype(
            'float32'), core_test_features.astype('float32')
        predict_on_song(seq_model, core_input_shape, mus_x_test, rec_x_test, core_test_features,
                        [musList[i]], [recList[i]], [matchesMapList[i]], i)

#note the last three args are assumed to be lists containing one element - the song we are currently
# predicting on. idk what happens if we try to predict on two songs at once
def predict_on_song(model, core_input_shape, mus_x_test, rec_x_test, core_test_features,
                    musList, recList, matchesMapList, songIndex):
    def to3d(arr):
        new = np.zeros((1, arr.shape[0], arr.shape[1]))
        new[0] = arr
        return new

    def to2d(arr):
        new = np.zeros((1, arr.shape[0]))
        new[0] = arr
        return new

    predictions = []
    originalMusStarts = []
    originalMusLens = []
    notePredictions = []

    for i in range(util.TIMESTEPS + 1):
        predictions.append(rec_x_test[0][i])
        notePredictions.append({
            'key': mus_x_test[0][i][2],
            'index':i, 'onv': mus_x_test[0][i][3],
            'offv':0, 'track':1,
            'offset': rec_x_test[0][i][0], 'len_offset':rec_x_test[0][i][1]
        })
        originalMusStarts.append(mus_x_test[0][i][0])
        originalMusLens.append(mus_x_test[0][i][1])

    for i in range(len(rec_x_test)):
        prev_predictions = to3d(rec_x_test[i])
        for j in range(util.TIMESTEPS + 1):
            prev_predictions[0][j] = predictions[-(util.TIMESTEPS+1) + j]
        # 0,1,2,3 is mus start, prev mus start, prev rec start, prev rec length, in chord
        originalMusStarts.append(core_test_features[i][0])
        originalMusLens.append(core_test_features[i][1])
        if core_input_shape >= 3:
            core_test_features[i][2] = predictions[-1][0]
        if core_input_shape >= 4:
            core_test_features[i][3] = predictions[-1][1]
        #print(to3d(mus_x_test[i]))
        #print(prev_predictions)
        pred = model.predict_on_batch([to3d(mus_x_test[i]), prev_predictions])
        predictions.append(pred[0])
        notePredictions.append({
            'key': mus_x_test[i][util.TIMESTEPS][2],
            'index': i + util.TIMESTEPS, 'onv': mus_x_test[i][util.TIMESTEPS][3],
            'offv': 0, 'track': 1,
            'offset': pred[0][0], 'len_offset': pred[0][1]
        })
        #print(pred[0])

    print(predictions)
    actual = []
    mus = []
    for musNote in musList[0]:
        musIndex = musNote['index']
        if musIndex in matchesMapList[0].keys():
            recNote = recList[0][matchesMapList[0][musIndex]]
            actual.append(recNote['start_normal'])
            mus.append(musNote['start_normal'])

    predictions = computeActualFromOffset(originalMusStarts, originalMusLens, predictions)

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

    lastTime = recList[0][-1]['start']
    predictions = util.denormalizeTimes(predictions, lastTime)
    assert(len(predictions) == len(notePredictions))
    for i in range(len(predictions)):
        notePredictions[i]['start'] = predictions[i][0]
        notePredictions[i]['end'] = predictions[i][0] + predictions[i][1]

    #print('lengths: rec_x_test = {}, predictions = {}, mus = {}'.format(len(rec_x_test), len(predictions), len(musList[0])))
    file = "C://Users//cpgaffney1//Documents//NetBeansProjects//ProjectMusic//files//predictions" + str(songIndex) + ".txt"
    with open(file, 'w') as of:
        '''for i in range(len(predictions) - 1):
            if i >= util.TIMESTEPS:
                of.write("{},{},{}\n".format(i, predictions[i+1][0], predictions[i+1][1]))
            else:
                of.write("{},{},{}\n".format(i, predictions[i][0], predictions[i][1]))'''
        for note in notePredictions:
            of.write(util.printNote(note) + '\n')



def main():
    ################
    oneModel = True  ##
    ################
    if oneModel:
        model = load_model('enhanced_nn_model.h5', custom_objects={'custom_loss': custom_loss})
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

        #print(model.get_layer("output").get_weights())
        predict(model, withOffset=True)


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()