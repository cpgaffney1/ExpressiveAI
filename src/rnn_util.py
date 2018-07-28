import os
import numpy as np
from src import note_util as note
from sklearn import preprocessing
from pathlib import Path

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
        files = None
        app = '_predict'
    else:
        app = ''

    musList = []
    musNames = []
    cwd = Path(os.getcwd())
    path = str(cwd.parent) + '/mus' + app
    musFiles = setFiles(files, path)
    for file in musFiles:
        name, song = note.readPIDI(file)
        musList.append(song)
        musNames.append(name)

    recList = []
    recNames = []
    path = str(cwd.parent) + '/rec' + app
    recFiles = setFiles(files, path)
    for file in recFiles:
        name, song = note.readPIDI(file)
        recList.append(song)
        recNames.append(name)

    assert(len(musList) == len(recList))
    return musList, musNames, recList, recNames

# counts as chord only if not the HEAD of a chord
def noteInChord(index, song):
    if index - 1 >= 0:
        if song[index - 1]['start'] == song[index]['start']:
            return True
    '''if index + 1 < len(song):
        if song[index + 1]['start'] == song[index]['start']:
            return True'''
    return False

def outputFeatureArray(musList, recList, bidirectional=False):
    songLengths = [len(r) for r in recList]
    if bidirectional:
        factor = 2
    else:
        factor = 1
    y = np.zeros((sum(songLengths) - len(recList) * factor * TIMESTEPS, 3))
    index = 0
    for i in range(len(recList)):
        for j in range(len(recList[i]) - factor * TIMESTEPS):
            y[index][0] = recList[i][j + TIMESTEPS]['offset_normal']
            y[index][1] = recList[i][j + TIMESTEPS]['len_offset_normal']
            y[index][2] = noteInChord(j + TIMESTEPS, musList[i])
            index += 1
    return y

def inputFeatureArray(musList, recList, bidirectional=False):
    if bidirectional:
        factor = 2
    else:
        factor = 1
    songLengths = [len(r) for r in musList]
    x_final = np.zeros((sum(songLengths) - len(musList) * factor * TIMESTEPS, factor * TIMESTEPS + 1, 6))
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
        end = x_final_index + len(musList[i]) - factor * TIMESTEPS
        take_indices = np.asarray([range(k, k + factor * TIMESTEPS + 1) for k in range(end - start)])
        assert (x_final[start: end].shape == (end - start, factor * TIMESTEPS + 1, 6))
        x_final[start : end] = x[take_indices]
        x_final[:, TIMESTEPS:, 4:6] = 0
        x_final_index += len(musList[i]) - factor * TIMESTEPS
    return x_final

def collectData(musList, recList, bidirectional=False):

    musList, recList = note.normalizeTimes(musList, recList)
    recList = note.addOffsets(musList, recList)

    y = outputFeatureArray(musList, recList, bidirectional=bidirectional)
    x = inputFeatureArray(musList, recList, bidirectional=bidirectional)

    x = x.astype('float32')
    y = y.astype('float32')
    return x, y


def load_data(files=None, for_predict=False, bidirectional=False):
    musList, musNames, recList, recNames = loadSongLists(files=files, for_predict=for_predict)
    global x_global
    global y_global
    x, y = collectData(musList, recList, bidirectional=bidirectional)
    x_global = x
    y_global = y
    return x, y
