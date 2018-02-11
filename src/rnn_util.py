import os
import numpy as np
from src import note_util as note


TIMESTEPS = 20

x_global = None
y_global = None

def setFiles(files, path):
    if files is None:
        files = [path + '/' + name for name in os.listdir(path)]
    else:
        files = [path + '/javaOutput' + str(i) + '.txt' for i in files]
    return files

# files is list of integers
def loadSongLists(files=None, for_predict=False):
    if for_predict:
        app = '_predict'
    else:
        app = ''

    musList = []
    musNames = []
    path = os.getcwd() + '/mus' + app
    musFiles = setFiles(files, path)
    for file in musFiles:
        name, song = note.readPIDI(file)
        musList.append(song)
        musNames.append(name)

    recList = []
    recNames = []
    path = os.getcwd() + '/rec' + app
    recFiles = setFiles(files, path)
    for file in recFiles:
        name, song = note.readPIDI(file)
        recList.append(song)
        recNames.append(name)

    assert(len(musList) == len(recList))
    for i in range(len(musList)):
        assert(len(musList[i]) == len(recList[i]))
    return musList, musNames, recList, recNames

def outputFeatureArray(recList):
    songLengths = [len(r) for r in recList]
    y = np.zeros((sum(songLengths) - len(recList) * 2 * TIMESTEPS, 2))
    index = 0
    for i in range(len(recList)):
        for j in range(len(recList[i]) - 2 * TIMESTEPS):
            y[index][0] = recList[i][j + TIMESTEPS]['offset_normal']
            y[index][1] = recList[i][j + TIMESTEPS]['len_offset_normal']
            index += 1
    return y

def inputFeatureArray(musList, recList):
    songLengths = [len(r) for r in musList]
    x_final = np.zeros((sum(songLengths) - len(musList) * 2 * TIMESTEPS, TIMESTEPS * 2 + 1, 6))
    x_final_index = 0
    for i in range(len(musList)):
        x = np.zeros((len(musList[i]), 6))
        for j in range(x.shape[0]):
            x[j][0] = musList[i][j]['start_normal']
            x[j][1] = musList[i][j]['end_normal']
            x[j][2] = musList[i][j]['key']
            x[j][3] = musList[i][j]['onv']
            x[j][4] = recList[i][j]['offset_normal']
            x[j][5] = recList[i][j]['len_offset_normal']

        start = x_final_index
        end = x_final_index + len(musList[i]) - 2 * TIMESTEPS
        take_indices = np.asarray([range(k, k + 2 * TIMESTEPS + 1) for k in range(end - start)])
        assert (x_final[start: end].shape == (end - start, 2 * TIMESTEPS + 1, 6))
        x_final[start : end] = x[take_indices]
        x_final[:, TIMESTEPS:, 4:6] = 0
        x_final_index += len(musList[i]) - 2 * TIMESTEPS
    return x_final

def collectData(musList, recList):
    recList = note.addOffsets(musList, recList)
    musList, recList = note.normalizeTimes(musList, recList)


    y = outputFeatureArray(recList)
    x = inputFeatureArray(musList, recList)

    x = x.astype('float32')
    y = y.astype('float32')
    return x, y


def load_data(files=None, for_predict=False):
    musList, musNames, recList, recNames = loadSongLists(files=files, for_predict=for_predict)
    global x_global
    global y_global
    x, y = collectData(musList, recList)
    x_global = x
    y_global = y
    return x, y
