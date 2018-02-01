from src import util
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K
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

def predict(model):
    musList, recList, matchesMapList, songNames = util.parseMatchedInput('testData', [0])
    musList, recList = util.normalizeTimes(musList, recList)
    recList, matchesMapList = util.trim(recList, matchesMapList)
    recList = util.addOffsets(musList, recList, matchesMapList)
    for i in range(len(musList)):
        x, y = util.dataAsWindowTwoSided([musList[i]], [recList[i]], [matchesMapList[i]])
        x_test = x.astype('float32')
        x_test = util.splitDataTwoSided(x_test, only_mus=True)
        x_test = x_test.astype('float32')
        predict_on_song_only_mus(model, x_test, [musList[i]], [recList[i]], [matchesMapList[i]], i)

#note the last three args are assumed to be lists containing one element - the song we are currently
# predicting on. idk what happens if we try to predict on two songs at once
def predict_on_song_output_feedback(model, x_test, musList, recList, matchesMapList, songIndex):
    def to3d(arr):
        new = np.zeros((1, arr.shape[0], arr.shape[1]))
        new[0] = arr
        return new

    predictions = []
    originalMusStarts = []
    originalMusLens = []
    notePredictions = []

    for i in range(util.TIMESTEPS + 1):
        predictions.append(x_test[0][i])
        notePredictions.append({
            'key': x_test[0][i][2],
            'index':i, 'onv': x_test[0][i][3],
            'offv':0, 'track':1,
            'offset': x_test[0][i][4], 'len_offset':x_test[0][i][5]
        })
        originalMusStarts.append(x_test[0][i][0])
        originalMusLens.append(x_test[0][i][1])

    for i in range(len(x_test)):
        prev_predictions = to3d(x_test[i])
        for j in range(util.TIMESTEPS + 1):
            prev_predictions[0][j] = predictions[-(util.TIMESTEPS + 1) + j]
        originalMusStarts.append(x_test[util.TIMESTEPS][0])
        originalMusLens.append(x_test[util.TIMESTEPS][1])
        pred = model.predict_on_batch([to3d(x_test[i]), prev_predictions])
        predictions.append(pred[0])
        notePredictions.append({
            'key': x_test[i][util.TIMESTEPS][2],
            'index': i + util.TIMESTEPS, 'onv': x_test[i][util.TIMESTEPS][3],
            'offv': 0, 'track': 1,
            'offset': pred[0][0], 'len_offset': pred[0][1]
        })

    organize_and_write_predictions(predictions, notePredictions, musList, recList, matchesMapList,
                                   originalMusStarts, originalMusLens, songIndex)


#note the last three args are assumed to be lists containing one element - the song we are currently
# predicting on. idk what happens if we try to predict on two songs at once
def predict_on_song_only_mus(model, x_test, musList, recList, matchesMapList, songIndex):
    def to3d(arr):
        new = np.zeros((1, arr.shape[0], arr.shape[1]))
        new[0] = arr
        return new

    predictions = []
    originalMusStarts = []
    originalMusLens = []
    notePredictions = []

    for i in range(util.TIMESTEPS + 1):
        predictions.append(x_test[0][i][:2])
        notePredictions.append({
            'key': x_test[0][i][2],
            'index':i, 'onv': x_test[0][i][3],
            'offv':0, 'track':1,
            'offset': 0, 'len_offset': 0
        })
        originalMusStarts.append(x_test[0][i][0])
        originalMusLens.append(x_test[0][i][1])

    print(x_test.shape[0])
    for i in range(x_test.shape[0]):
        originalMusStarts.append(x_test[i][util.TIMESTEPS][0])
        originalMusLens.append(x_test[i][util.TIMESTEPS][1])
        pred = model.predict_on_batch(to3d(x_test[i]))
        predictions.append(pred[0])
        notePredictions.append({
            'key': x_test[i][util.TIMESTEPS][2],
            'index': i + util.TIMESTEPS, 'onv': x_test[i][util.TIMESTEPS][3],
            'offv': 0, 'track': 1,
            'offset': pred[0][0], 'len_offset': pred[0][1]
        })

    organize_and_write_predictions(predictions, notePredictions, musList, recList, matchesMapList,
                                       originalMusStarts, originalMusLens, songIndex)

def organize_and_write_predictions(predictions, notePredictions, musList, recList, matchesMapList,
                                   originalMusStarts, originalMusLens, songIndex):
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
    print(predictions)
    start_pred = []
    for i in range(len(predictions)):
        start_pred.append(predictions[i][0])
    plt.plot(actual, label='actual')
    plt.plot(mus, label='input')
    print(start_pred)
    plt.plot(start_pred, label='predictions')
    print()
    actual = actual[:len(actual) - (len(actual) - len(start_pred))]
    sq = [(actual[i] - start_pred[i]) ** 2 for i in range(len(actual))]
    print(np.mean(np.asarray(sq)))
    plt.legend()
    plt.show()

    lastTime = recList[0][-1]['start']
    predictions = util.denormalizeTimes(predictions, lastTime)
    assert (len(predictions) == len(notePredictions))
    for i in range(len(predictions)):
        notePredictions[i]['start'] = predictions[i][0]
        notePredictions[i]['end'] = predictions[i][0] + predictions[i][1]

    # print('lengths: rec_x_test = {}, predictions = {}, mus = {}'.format(len(rec_x_test), len(predictions), len(musList[0])))
    file = "C://Users//cpgaf//OneDrive//Documents//NetBeansProjects//Expressive//files" + str(
        songIndex) + ".txt"
    with open(file, 'w') as of:
        '''for i in range(len(predictions) - 1):
            if i >= util.TIMESTEPS:
                of.write("{},{},{}\n".format(i, predictions[i+1][0], predictions[i+1][1]))
            else:
                of.write("{},{},{}\n".format(i, predictions[i][0], predictions[i][1]))'''
        for note in notePredictions:
            of.write(util.printNote(note) + '\n')

def main():
    model = load_model("rnn_model.h5")
    predict(model)


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()

