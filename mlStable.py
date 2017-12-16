import util
import heapq
import copy
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# x features
FEATURES = ['mus start', 'prev rec start', 'mus key', 'mus onv',
            'moving average start delta', #'long term average start delta',
            'moving average key', 'moving average onv',
            #'mus note length'
            ]
# y output = predicted tick
N_FEATURES = len(FEATURES)
FRAME = 20

def predict(testMus, weights, intercept, FRAME):
    key = testMus[0]['key']
    onv = testMus[0]['onv']
    frameSize = 1
    movingSumDelta = 0
    totalSumDelta = 0
    movingSumKey = key
    movingSumOnv = onv
    removeI = 0
    prevPrediction = 0
    y = [prevPrediction]
    noteCount = 1
    for i in range(1, len(testMus)):
        delta = y[i-1] - testMus[i-1]['start']
        if i < FRAME:
            frameSize += 1
            movingSumDelta += delta
            totalSumDelta += delta
            movingSumKey += testMus[i]['key']
            movingSumOnv += testMus[i]['onv']
        else:
            assert(frameSize == FRAME)
            removeNote = testMus[removeI]
            deltaRemove = y[removeI] - removeNote['start']
            movingSumDelta += delta - deltaRemove
            totalSumDelta += delta
            movingSumKey += testMus[i]['key'] - removeNote['key']
            movingSumOnv += testMus[i]['onv'] - removeNote['onv']
            removeI += 1
        x = [testMus[i]['start'], prevPrediction, testMus[i]['key'],testMus[i]['onv'],
             movingSumDelta/frameSize, #totalSumDelta/noteCount,
             movingSumKey/frameSize,movingSumOnv/frameSize,
             #testMus[i]['end'] - testMus[i]['start']
             ]
        y.append(np.dot(x, weights) + intercept)
        print(x)
        prevPrediction = y[-1]
        noteCount += 1
    return y

def aggregateSongData(data, mus, rec, matchesMap):
    movingSumDelta = 0
    totalSumDelta = 0
    movingSumKey = 0
    movingSumOnv = 0
    removeMusI = 0
    noteCount = 1
    prevRec = rec[matchesMap[0]]
    prevMus = mus[0]
    removeRecI = matchesMap[removeMusI]
    frameSize = 0
    for i in range(len(mus)):
        assert(frameSize <= FRAME)
        # mus index i was not matched
        if i not in matchesMap:
            continue
        musNote = mus[i]
        recNote = rec[matchesMap[i]]
        delta = prevRec['start'] - prevMus['start']
        if i < FRAME:
            frameSize += 1
            movingSumDelta += delta
            totalSumDelta += delta
            movingSumKey += musNote['key']
            movingSumOnv += musNote['onv']
        else:
            removeMusNote = mus[removeMusI]
            removeRecNote = rec[removeRecI]
            deltaRemove = removeRecNote['start'] - removeMusNote['start']
            movingSumDelta += delta - deltaRemove
            totalSumDelta += delta
            movingSumKey += musNote['key'] - removeMusNote['key']
            movingSumOnv += musNote['onv'] - removeMusNote['onv']
            if frameSize >= FRAME:
                removeMusI += 1
            else:
                frameSize += 1
            if tryToIncrementFrame(removeMusI, matchesMap, frameSize):
                removeMusI, frameSize = tryToIncrementFrame(removeMusI, matchesMap, frameSize, False)
            removeRecI = matchesMap[removeMusI]
        newRow = [musNote['start'], prevRec['start'], musNote['key'], musNote['onv'],
                  movingSumDelta/frameSize, #totalSumDelta/noteCount,
                  movingSumKey/frameSize, movingSumOnv/frameSize,
                  #musNote['end'] - musNote['start'],
                  recNote['start']
                  ]
        data = np.vstack([data, newRow])
        print(newRow)
        print(frameSize)
        prevRec = recNote
        prevMus = musNote
        noteCount += 1
    return data

def tryToIncrementFrame(removeMusI, matchesMap, frameSize, attempt=True):
    while removeMusI not in matchesMap:
        removeMusI += 1
        frameSize -= 1
    if frameSize <= 0:
        if attempt:
            return False
        else:
            return -1, -1
    else:
        if attempt:
            return True
        else:
            return removeMusI, frameSize

def performMatching(mus, rec, musPreferences, recPreferences):
    for i in musPreferences:
        pq = musPreferences[i]
        lis = []
        while True:
            min = pq.removeMin()
            if min == (None, None): break
            lis.append(min[0])
        musPreferences[i] = lis

    for i in recPreferences:
        pq = recPreferences[i]
        lis = []
        while True:
            min = pq.removeMin()
            if min == (None, None): break
            lis.append(min[0])
            recPreferences[i] = lis

    matchesMap = {}
    matches = findMatches(len(mus), len(rec), musPreferences, recPreferences)
    for m in matches:
        matchesMap[m[0]] = m[1]
    #print(matchesMap)
    #print('  ' + ',\n  '.join('Mus %s is matched to rec %s' % m for m in matches))

    with open("C:\\Users\\cpgaffney1\\Documents\\NetBeansProjects\\ProjectMusic\\files\\matchedNotes.txt", 'w') as of:
        for pair in matches:
            of.write("{},{}\n".format(pair[0],pair[1]))

    return matchesMap

def findMatches(musLen, recLen, musPreferences, recPreferences):
    #list of (mus, rec)
    matches = []
    musFree = [i for i in range(musLen)]
    # a certain proportion will remain unmatched
    endLength = len(musFree) * 0.01
    recFree = [i for i in range(recLen)]
    while len(musFree) > endLength:
        currMus = musFree[0]
        # all his choices are gone, so sad. He gets no match
        if len(musPreferences[currMus]) == 0:
            musFree.pop(0)
            continue
        recChoice = musPreferences[currMus][0]
        if recChoice in recFree:
            matches.append((currMus, recChoice))
            musFree.pop(0)
            recFree.remove(recChoice)
        else:
            currMatch = findRecPair(matches, recChoice)
            if currMatch == None:
                print(recChoice)
            # rec would prefer currMus over its current partner
            if recPreferences[recChoice].index(currMus) < recPreferences[recChoice].index(currMatch[0]):
                musFree.pop(0)
                musFree.append(currMatch[0])
                matches.remove(currMatch)
                matches.append((currMus, recChoice))
        musPreferences[currMus].pop(0)
    return matches


def findRecPair(matches, recNote):
    for m in matches:
        if m[1] == recNote:
            return m

def writeOutput(ucs, mus, rec):
    output = "matched.csv"
    with open(output) as of:
        of.write("key,musIndex,onv,offv,musStart,musEnd,recIndex,recStart,recEnd\n")
        for a in ucs.actions:
            musNote = mus[a[0]]
            recNote = rec[a[1]]
            of.write(printNoteInCSV(musNote) + ',')
            of.write('{},{},{}'.format(recNote['index'], recNote['start'], recNote['end']))
            of.write('\n')

def printNoteInCSV(note):
    ret = ""
    for key, value in note:
        ret += '{},'.format(value)
    ret = ret[:-1]
    return ret

def decodeNote(str):
    note = {}
    arr = str.split(',')
    note['key'] = int(arr[0])
    note['index'] = int(arr[1])
    note['onv'] = int(arr[2])
    note['offv'] = int(arr[3])
    note['start'] = int(arr[4])
    note['end'] = int(arr[5])
    return note

def printNote(note):
    return "{{{},{},{},{},{},{}}}".format(note['key'], note['index'],note['onv'],note['offv'],
        note['start'],note['end'])


def main():
    file = "javaOutput.txt"
    with open(file) as f:
        lines = f.readlines()
    songCount = 0

    #initialize storage variables
    #list of rec notes
    rec = []
    # list of mus notes
    mus = []
    #men
    musPreferences = {}
    #women
    recPreferences = {}
    #dataset for training
    data = np.zeros((0, N_FEATURES + 1))
    parsingMatches = True

    # see java code for input format
    for line in lines:
        #starting to read data from new song
        if line == '---\n':
            print('new song')
            songCount += 1
            matchesMap = performMatching(mus, rec, musPreferences, recPreferences)
            data = aggregateSongData(data, mus, rec, matchesMap)
            # reset variables
            # list of rec notes
            rec = []
            # list of mus notes
            mus = []
            # men
            musPreferences = {}
            # women
            recPreferences = {}
            parsingMatches = True
        #starting to read rec data associated with same song
        elif line == '***\n':
            parsingMatches = False
        #continuing to read matching candidate data
        elif parsingMatches:
            arr = line.split(":")
            musNote = decodeNote(arr[0][1:-1])
            musIndex = musNote['index']
            mus.append(musNote)
            musPreferences[musIndex] = util.PriorityQueue()
            arr = arr[1].split(';')
            for str in arr:
                recNote, lcs, percent, distance = str.split('/')
                recNote = decodeNote(recNote[1:-1])
                recIndex = recNote['index']
                lcs = float(lcs)
                percent = float(percent)
                distance = float(distance)
                # calculate cost. smaller cost is better
                cost = 1/3 * ((1 - lcs) + (1 - percent) + distance)
                musPreferences[musIndex].update(recIndex, cost)
                if recIndex not in recPreferences:
                    recPreferences[recIndex] = util.PriorityQueue()
                recPreferences[recIndex].update(musIndex, cost)
        #parsing rec notes
        elif not parsingMatches:
            recNote = decodeNote(line[1:-2])
            rec.append(recNote)
        else: assert(False)

    reg = linear_model.LinearRegression()
    reg.fit(data[:,:N_FEATURES],data[:,N_FEATURES])

    file = "testMus.txt"
    with open(file) as f:
        lines = f.readlines()
    testMus = []
    # see java code for input format
    for line in lines:
        note = decodeNote(line[1:-2])
        testMus.append(note)

    prediction = predict(testMus, reg.coef_, reg.intercept_, FRAME)

    file = "C:\\Users\\cpgaffney1\\Documents\\NetBeansProjects\\ProjectMusic\\files\\predictions.txt"
    with open(file, 'w') as of:
        for i in range(len(testMus)):
            of.write("{},{}\n".format(i,prediction[i]))

    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(data[:, :N_FEATURES], data[:, N_FEATURES]))

    actual = []
    for note in testMus:
        actual.append(note['start'])
    plt.plot(actual)
    plt.plot(prediction)
    plt.show()


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()