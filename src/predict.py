import util
import math
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import pandas as pd


# x features
features = ['mus start', #'mus end',
            'mus key', 'mus onv', #'mus track',
            'prev start delta',
            'prev rec start',
            'moving average start delta',
            'long term average start delta',
            'moving average key', 'moving average onv',
            'mus note length'
            ]

# y output = predicted tick
FRAME = 20

# if rec == None, then we're predicting
def aggregateOrPredict(mus, rec, matchesMap, startWeights, predict=False):
    # if we are predicting
    if predict:
        indices = range(len(mus))
        iterator = indices
    else:
        indices = range(len(matchesMap))
        iterator = sorted(matchesMap.keys())
    x = pd.DataFrame(columns=features, index=indices)
    y = pd.DataFrame(columns=["mus index", "rec start"], index=indices)
    noteCount = 1
    for i in iterator:
        musNote = mus[i]
        if noteCount == 1:
            if predict:
                x.iloc[noteCount - 1] = [musNote['start'], #musNote['end'],
                                         musNote['key'], musNote['onv'], #musNote['track'],
                                         0,
                                         0,
                                         0,
                                         0,
                                         musNote['key'], musNote['onv'],
                                         musNote['end'] - musNote['start']
                                         ]
            else:
                recNote = rec[matchesMap[i]]
                startDelta = recNote['start'] - musNote['start']
                if noteCount == 1:
                    x.iloc[noteCount - 1] = [musNote['start'], #musNote['end'],
                                             musNote['key'], musNote['onv'], #musNote['track'],
                                             startDelta,
                                             recNote['start'],
                                             startDelta,
                                             startDelta,
                                             musNote['key'], musNote['onv'],
                                             musNote['end'] - musNote['start']
                                             ]
        else: #not on first elementp
            if i % 100 == 0: print(i)
            movingXframe = x.iloc[max(0, noteCount - FRAME): max(1, noteCount - 2)]
            prevStartDelta = y.get_value(noteCount - 2, 'rec start') - x.get_value(noteCount - 2, 'mus start')
            x.iloc[noteCount - 1] = [musNote['start'], #musNote['end'],
                                     musNote['key'], musNote['onv'], #musNote['track'],
                                     prevStartDelta,
                                     y.get_value(noteCount - 2, 'rec start'),
                                     movingXframe['prev start delta'].mean(),
                                     (x.get_value(noteCount - 2, 'long term average start delta') * (noteCount - 1) + prevStartDelta) / noteCount,
                                     movingXframe['mus key'].mean(), movingXframe['mus onv'].mean(),
                                     musNote['end'] - musNote['start']
                                     ]
        if predict:
            startPred = np.dot(x.iloc[noteCount - 1].values, startWeights[0]) + startWeights[1]
            if noteCount < 20:
                y.iloc[noteCount - 1] = [i, musNote['start']]
            else:
                y.iloc[noteCount - 1] = [i, startPred]
        else:
            recNote = rec[matchesMap[i]]
            y.iloc[noteCount - 1] = [i, recNote['start']]
        noteCount += 1
    print(x)
    return x, y

def main():
    dataList, _, _, _= parseMatchedInput('javaOutput.txt')
    testPair, testMus, testRec, matchesMap = parseMatchedInput('testData.txt')

    print('training')
    x, y = dataList[0]
    train = x.values
    result = y['rec start'].values
    for i in range(1, len(dataList)):
        x, y = dataList[i]
        train = np.concatenate((train, x))
        result = np.concatenate((result, y['rec start']))
    startReg = linear_model.LinearRegression()
    startReg.fit(train, result)
    print('done training')

    x, y = aggregateOrPredict(testMus, None, None, (startReg.coef_, startReg.intercept_), predict=True)

    assert(len(matchesMap) <= y.shape[0])
    sqSum = 0
    for i in matchesMap.keys():
        sqSum += (y[y['mus index'] == i]['rec start'].values[0] - testRec[matchesMap[i]]['start']) ** 2
    norm = math.sqrt(sqSum)

    print('Vector distance between prediction and recording = {}'.format(norm))

    prediction = []
    file = "C:\\Users\\cpgaffney1\\Documents\\NetBeansProjects\\ProjectMusic\\files\\predictions.txt"
    with open(file, 'w') as of:
        for i in range(len(testMus)):
            prediction.append(y.iloc[i]['rec start'])
            of.write("{},{}\n".format(i, y.iloc[i]['rec start']))

    print(startReg.coef_)
    print(startReg.intercept_)
    print(startReg.score(x, y['rec start']))

    actual = []
    for note in testMus:
        actual.append(note['start'])
    plt.plot(actual)
    plt.plot(prediction)
    plt.show()


















def parseMatchedInput(filename):
    file = filename
    with open(file) as f:
        lines = f.readlines()
    songCount = 0

    # initialize storage variables
    # list of x, y pairs corresponding to song
    dataList = []
    # list of rec notes
    rec = []
    # list of mus notes
    mus = []

    lastRec = []
    lastMus = []
    # men
    musPreferences = {}
    # women
    recPreferences = {}
    parsingMatches = True

    # see java code for input format
    for line in lines:
        # starting to read data from new song
        if line == '---\n':
            print('new song')
            songCount += 1
            print('matching')
            matchesMap = performMatching(mus, rec, musPreferences, recPreferences)
            print('done matching, starting aggregate')
            x, y = aggregateOrPredict(mus, rec, matchesMap, None)
            print('done aggregating')
            dataList.append((x, y))
            # reset variables
            lastRec = rec.copy()
            lastMus = mus.copy()
            # list of rec notes
            rec = []
            # list of mus notes
            mus = []
            # men
            musPreferences = {}
            # women
            recPreferences = {}
            parsingMatches = True
        # starting to read rec data associated with same song
        elif line == '***\n':
            parsingMatches = False
        # continuing to read matching candidate data
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
                cost = 1 / 3 * ((1 - lcs) + (1 - percent) + distance)
                musPreferences[musIndex].update(recIndex, cost)
                if recIndex not in recPreferences:
                    recPreferences[recIndex] = util.PriorityQueue()
                recPreferences[recIndex].update(musIndex, cost)
        # parsing rec notes
        elif not parsingMatches:
            recNote = decodeNote(line[1:-2])
            rec.append(recNote)
        else:
            assert (False)
    assert(matchesMap != None)
    return dataList, lastMus, lastRec, matchesMap

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


def decodeNote(str):
    note = {}
    arr = str.split(',')
    note['key'] = int(arr[0])
    note['index'] = int(arr[1])
    note['onv'] = int(arr[2])
    note['offv'] = int(arr[3])
    note['start'] = int(arr[4])
    note['end'] = int(arr[5])
    note['track'] = int(arr[6])
    return note

def printNote(note):
    return "{{{},{},{},{},{},{}}}".format(note['key'], note['index'],note['onv'],note['offv'],
        note['start'],note['end'],note['track'])

# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()