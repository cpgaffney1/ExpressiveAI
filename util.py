import heapq, collections, re, sys, time, os, random
import numpy as np

#########################################################
# util functions for music

#note that midi has 128 notes, more than the 88 on a keyboard
KEYBOARD = 128
TIMESTEPS = 20

H_LAYER_SIZE = KEYBOARD
DATA_LEN = 32554
NUM_BATCHES = 82
BATCH_SIZE = DATA_LEN // NUM_BATCHES#H_LAYER_SIZE * 2
N_EPOCHS = 1000
NON_SPARSE_FEATURES = (TIMESTEPS + 1) * 6 + TIMESTEPS * 4
SIMPLIFIED_FEATURES = 6
V_SIMPLE_FEATURES = 2

def trim(recList, matchesMapList):
    for i in range(len(recList)):
        window_length = 5
        j = 0
        window = []
        while j < len(matchesMapList[i]):
            keys = list(sorted(matchesMapList[i].keys()))
            musIndex = keys[j]
            recIndex = matchesMapList[i][musIndex]
            recNote = recList[i][recIndex]
            window.append(recNote['start_normal'])
            moving_avg = sum(window) / len(window)
            if abs(recNote['start_normal'] - moving_avg) > 4:
                print(moving_avg)
                del recList[i][recIndex]
                del matchesMapList[i][musIndex]
                r, m = refactor(j, recList[i], matchesMapList[i])
                recList[i] = r
                matchesMapList[i] = m
                j -= 1
            if j >= window_length - 1:
                del window[0]
            j += 1
        #print(recList[i])
    return recList, matchesMapList

def refactor(removedIndex, rec, matches):
    print("refactoring")
    for note in rec:
        if note['index'] > removedIndex:
            note['index'] -= 1
    for musIndex in matches.keys():
        recIndex = matches[musIndex]
        if recIndex > removedIndex:
            matches[musIndex] -= 1
    return rec, matches


def dataAsWindow(musList, recList, matchesMapList, simplified=False, verySimple=False):
    if verySimple: simplified = True
    print('transforming')
    songLengths = [len(mus.keys()) for mus in matchesMapList]
    numObs = sum(songLengths) - len(matchesMapList) * (2 * TIMESTEPS + 1) - len(matchesMapList)
    print(numObs)
    if verySimple:
        dim = V_SIMPLE_FEATURES
    elif simplified:
        dim = SIMPLIFIED_FEATURES
    else:
        dim = NON_SPARSE_FEATURES
    x = np.zeros((numObs, dim))
    y = np.zeros((numObs, 2))
    obs_index = 0
    for song_num in range(len(musList)):
        print(song_num)
        # list of mus_start, mus_end, key, onv, rec_start, rec_end for each timestep
        mus = musList[song_num]
        rec = recList[song_num]
        match = matchesMapList[song_num]
        keys = sorted(match.keys())
        for i in range(TIMESTEPS + 1, len(keys) - TIMESTEPS - 1):
            x_obs = []
            prevRecIndex = match[keys[i - TIMESTEPS - 1]]
            slice = keys[i - TIMESTEPS - 1: i + TIMESTEPS]
            for k in range(len(slice)):
                musIndex = slice[k]
                if simplified and musIndex != keys[i]:
                    continue
                x_obs.append(mus[musIndex]['start_normal'])
                assert(mus[musIndex]['end_normal'] - mus[musIndex]['start_normal'] >= 0)
                if not verySimple: x_obs.append(mus[musIndex]['end_normal'] - mus[musIndex]['start_normal'])
                if not verySimple: x_obs.append(mus[musIndex]['key'])
                if not verySimple: x_obs.append(mus[musIndex]['onv'])
                if k < TIMESTEPS + 1:
                    #if not verySimple:
                    x_obs.append(rec[prevRecIndex]['offset'])
                    #x_obs.append(rec[prevRecIndex]['end_normal'] - rec[prevRecIndex]['start_normal'])
                    x_obs.append(rec[prevRecIndex]['len_offset'])
                    assert(rec[prevRecIndex]['end_normal'] - rec[prevRecIndex]['start_normal'] >= 0)
                prevRecIndex = match[musIndex]

            x[obs_index] = np.asarray(x_obs)
            recNote = rec[match[keys[i]]]
            #y[obs_index] = [recNote['offset'], recNote['end_normal'] - recNote['start_normal']]
            y[obs_index] = [recNote['offset'], recNote['len_offset']]
            assert(rec[match[keys[i]]]['end_normal'] - recNote['start_normal']) >= 0
            obs_index += 1
    assert(len(x) == len(y))
    return x, y

'''
def dataAsWindowForPredict(mus, rec, match, simplified=False, verySimple=False):
    if verySimple: simplified = True
    print('transforming')
    musNoteIndices = []
    numObs = len(mus) - (2 * TIMESTEPS + 1) - 1
    if verySimple:
        dim = V_SIMPLE_FEATURES
    elif simplified:
        dim = SIMPLIFIED_FEATURES
    else:
        dim = NON_SPARSE_FEATURES
    x = np.zeros((numObs, dim))
    obs_index = 0
    initial_predictions = []
    for i in range(TIMESTEPS + 1, len(mus) - TIMESTEPS - 1):
        x_obs = []
        slice = mus[i - TIMESTEPS : i + TIMESTEPS + 1]
        for k in range(len(slice)):
            musIndex = slice[k]['index']
            if simplified and mus[i]['index'] != musIndex:
                continue
            x_obs.append(mus[musIndex]['start_normal'])
            if not verySimple: x_obs.append(mus[musIndex]['end_normal'] - mus[musIndex]['start_normal'])
            if not verySimple: x_obs.append(mus[musIndex]['key'])
            if not verySimple: x_obs.append(mus[musIndex]['onv'])
            if k < TIMESTEPS + 1:
                if obs_index == 0:
                    prevRecIndex = match[slice[k - 1]['index']]
                    if verySimple:
                        pair = (rec[prevRecIndex]['end_normal'],)
                    else:
                        pair = (rec[prevRecIndex]['start_normal'], rec[prevRecIndex]['end_normal'])
                    initial_predictions.append(pair)
                if not verySimple: x_obs.append(-1)
                x_obs.append(-1)
        x[obs_index] = np.asarray(x_obs)
        musNoteIndices.append(mus[i]['index'])
        obs_index += 1
    print('done transforming')
    return x, initial_predictions, musNoteIndices
'''

'''
def dataAsSequential(musList, recList, matchesMapList):
    print('transforming')
    print(musList[0][0])
    xTrainList = []
    yTrainList = []
    numObs = 0
    countWeirdOccurences = 0
    songLengths = []
    for i in range(len(musList)):
        musTrain = []
        recTrain = []
        mus = musList[i]
        rec = recList[i]
        match = matchesMapList[i]
        lastStartTime = -1
        x = None
        y = None
        indices = None
        for musIndex in sorted(match.keys()):
            musNote = mus[musIndex]
            recNote = rec[match[musIndex]]
            if musNote['start_normal'] != lastStartTime:
                if x is not None and y is not None:
                    musTrain.append(x)
                    recTrain.append(y)
                lastStartTime = musNote['start_normal']
                x = np.zeros(KEYBOARD * 3 + 1)
                indices = [-1 for _ in range(KEYBOARD)]
                y = np.zeros(KEYBOARD)
            if x[musNote['key']] != 0: countWeirdOccurences += 1
            assert(musNote['start_normal'] == lastStartTime)
            x[musNote['key']] = 1
            indices[musNote['key']] = musNote['index']
            x[KEYBOARD + musNote['key']] = musNote['onv']
            x[2 * KEYBOARD + musNote['key']] = musNote['end_normal']
            x[-1] = lastStartTime
            assert(recNote['key'] == musNote['key'])
            y[recNote['key']] = recNote['start_normal']
        xTrainList.append(musTrain)
        numObs += len(musTrain) + 1 - TIMESTEPS
        yTrainList.append(recTrain)
    print('weird occurences ' + str(countWeirdOccurences))
    print('num obs ' + str(numObs))
    x = np.zeros((numObs, TIMESTEPS, KEYBOARD * 3 + 1))
    y = np.zeros((numObs, KEYBOARD))
    obsIndex = 0
    for i in range(len(xTrainList)):
        songLengths.append(0)
        musTrain = xTrainList[i]
        recTrain = yTrainList[i]
        for j in range(TIMESTEPS - 1, len(musTrain)):
            songLengths[-1] += 1
            for k in range(TIMESTEPS):
                x[obsIndex][k] = musTrain[j - (TIMESTEPS - k) + 1]
                y[obsIndex] = recTrain[j]
            obsIndex += 1
    print('done transforming')
    assert(len(songLengths) == len(musList))
    return x, y, songLengths

def splitSeqChord(mus_x_train, rec_x_train, core_train_features, y_train, isSeq):
    isChordIndices = []
    notChordIndices = []
    for i in range(len(core_train_features)):
        if core_train_features[i][0] == core_train_features[i][1]:
            isChordIndices.append(i)
        else:
            notChordIndices.append(i)

    mus_train_seq = mus_x_train[notChordIndices][:][:]
    rec_train_seq = rec_x_train[notChordIndices][:][:]
    core_train_seq = core_train_features[notChordIndices][:]
    y_train_seq = y_train[notChordIndices][:]

    mus_train_chord = mus_x_train[isChordIndices][:][:]
    rec_train_chord = rec_x_train[isChordIndices][:][:]
    core_train_chord = core_train_features[isChordIndices][:]
    y_train_chord = y_train[isChordIndices][:]

    if isSeq:
        return mus_train_seq, rec_train_seq, core_train_seq, y_train_seq
    else:
        return mus_train_chord, rec_train_chord, core_train_chord, y_train_chord
'''
def splitData(x, core_input_size=5):
    assert(core_input_size >= 1 and core_input_size <= 5)
    mus_x_train = np.zeros((len(x), 2 * TIMESTEPS + 1, 4))
    rec_x_train = np.zeros((len(x), TIMESTEPS + 1, 2))
    core_train_features = np.zeros((len(x), core_input_size))
    for i in range(len(x)):
        # mus start, mus length, prev rec start, prev rec length, in chord
        core_train_features[i][0] = x[i][TIMESTEPS * 6 + 0]
        if core_input_size >= 2:
            core_train_features[i][1] = x[i][(TIMESTEPS) * 6 + 1]
        if core_input_size >= 3:
            core_train_features[i][2] = x[i][TIMESTEPS * 6 + 4]
        if core_input_size >= 4:
            core_train_features[i][3] = x[i][TIMESTEPS * 6 + 5]
        if core_input_size >= 5:
            core_train_features[i][4] = core_train_features[i][0] == core_train_features[i][1]
        rec = np.zeros((TIMESTEPS + 1, 2))
        for t in range(len(rec)):
            rec[t][0] = x[i][t * 6 + 4]
            rec[t][1] = x[i][t * 6 + 5]
        mus = np.zeros((2 * TIMESTEPS + 1, 4))
        for t in range(len(mus)):
            for j in range(len(mus[t])):
                offset = (t * 6) if t<=(TIMESTEPS+1) else ((TIMESTEPS+1)*6 + (t-(TIMESTEPS+1))*4)
                mus[t][j] = x[i][offset + j]
        mus_x_train[i] = mus
        rec_x_train[i] = rec
    return mus_x_train, rec_x_train, core_train_features

'''
def dataAsSequentialForPredict(mus, matches):
    print('transforming')
    numObs = 0
    countWeirdOccurences = 0
    for i in range(len(musList)):
        musTrain = []
        recTrain = []
        mus = musList[i]
        rec = recList[i]
        match = matchesMapList[i]
        lastStartTime = -1
        x = None
        y = None
        indices = None
        for musIndex in sorted(match.keys()):
            musNote = mus[musIndex]
            recNote = rec[match[musIndex]]
            if musNote['start_normal'] != lastStartTime:
                if x is not None and y is not None:
                    musTrain.append(x)
                    recTrain.append(y)
                lastStartTime = musNote['start_normal']
                x = np.zeros(KEYBOARD * 3 + 1)
                indices = [-1 for _ in range(KEYBOARD)]
                y = np.zeros(KEYBOARD)
            if x[musNote['key']] != 0: countWeirdOccurences += 1
            assert(musNote['start_normal'] == lastStartTime)
            x[musNote['key']] = 1
            indices[musNote['key']] = musNote['index']
            x[KEYBOARD + musNote['key']] = musNote['onv']
            x[2 * KEYBOARD + musNote['key']] = musNote['end_normal']
            x[-1] = lastStartTime
            assert(recNote['key'] == musNote['key'])
            y[recNote['key']] = recNote['start_normal']
        xTrainList.append(musTrain)
        numObs += len(musTrain) + 1 - TIMESTEPS
        yTrainList.append(recTrain)
    print('weird occurences ' + str(countWeirdOccurences))
    print('num obs ' + str(numObs))
    x = np.zeros((numObs, TIMESTEPS, KEYBOARD * 3 + 1))
    y = np.zeros((numObs, KEYBOARD))
    obsIndex = 0
    for i in range(len(xTrainList)):
        songLengths.append(0)
        musTrain = xTrainList[i]
        recTrain = yTrainList[i]
        for j in range(TIMESTEPS - 1, len(musTrain)):
            songLengths[-1] += 1
            for k in range(TIMESTEPS):
                x[obsIndex][k] = musTrain[j - (TIMESTEPS - k) + 1]
                y[obsIndex] = recTrain[j]
            obsIndex += 1
    print('done transforming')
    assert(len(songLengths) == len(musList))
    return x, y, songLengths '''

def load_data(core_input_shape=5, n_files=252):
    musList, recList, matchesMapList, songNames = parseMatchedInput('javaOutput/javaOutput', range(0,n_files))
    musList, recList = normalizeTimes(musList, recList)
    recList, matchesMapList = trim(recList, matchesMapList)
    recList = addOffsets(musList, recList, matchesMapList)
    x, y = dataAsWindow(musList, recList, matchesMapList)
    x_train = x.astype('float32')
    y_train = y.astype('float32')
    mus_x_train, rec_x_train, core_train_features = splitData(x_train, core_input_size=core_input_shape)
    mus_x_train, rec_x_train, core_train_features = mus_x_train.astype('float32'), rec_x_train.astype(
        'float32'), core_train_features.astype('float32')
    return mus_x_train, rec_x_train, core_train_features, y_train

def denormalizeTimes(predictions, lastTime):
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            predictions[i][j] *= (lastTime / 100.0)
    return predictions

# sets starting and ending times to be percentage of total song length
def normalizeTimes(musList, recList):
    for i in range(len(musList)):
        lastTime = float(musList[i][-1]['start'])
        for note in musList[i]:
            note['start_normal'] = (note['start'] * 100) / lastTime
            note['end_normal'] = (note['end'] * 100) / lastTime
    for i in range(len(recList)):
        lastTime = float(recList[i][-1]['start'])
        for note in recList[i]:
            note['start_normal'] = (note['start'] * 100) / lastTime
            note['end_normal'] = (note['end'] * 100) / lastTime
    return musList, recList

def addOffsets(musList, recList, matchesMapList):
    for i in range(len(musList)):
        mus = musList[i]
        rec = recList[i]
        match = matchesMapList[i]
        keys = sorted(match.keys())
        for mIndex in keys:
            m = mus[mIndex]
            r = rec[match[mIndex]]
            r['offset'] = r['start_normal'] - m['start_normal']
            r['len_offset'] = (r['end_normal'] - r['start_normal']) - (m['end_normal'] - m['start_normal'])
            recList[i][match[mIndex]] = r
    return recList

def convertToStatefulBatched(x_train, y_train):
    x_new_shape = (BATCH_SIZE * (len(x_train) - 1), x_train.shape[1], x_train.shape[2])
    y_new_shape = (BATCH_SIZE * (len(y_train) - 1), y_train.shape[1])
    print(x_new_shape)
    x_new = np.zeros(x_new_shape)
    y_new = np.zeros(y_new_shape)
    x_batch_list = []
    y_batch_list = []
    for i in range(len(x_train) - BATCH_SIZE + 1):
        x_new[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = x_train[i:BATCH_SIZE]
        y_new[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = y_train[i:BATCH_SIZE]
        #x_batch_list.append(x_train[i:rnn.BATCH_SIZE])
        #y_batch_list.append(y_train[i:rnn.BATCH_SIZE])
    print(x_new)
    print(y_new)
    return x_new, y_new

def parseMatchedInput(filename, fileNums, exclude=[]):
    musList = []
    recList = []
    matchesMapList = []
    songNames = []
    for i in fileNums:
        if i in exclude:
            continue
        ml, rl, mml, sn = parseMatchedInputFromFile(filename + str(i) + '.txt')
        musList += ml
        recList += rl
        matchesMapList += mml
        songNames += sn
    return musList, recList, matchesMapList, songNames


def parseMatchedInputFromFile(filename):
    file = filename
    with open(file) as f:
        lines = f.readlines()
    songCount = 0

    # initialize storage variables
    matchesMapList = []
    musList = []
    recList = []
    # list of rec notes
    rec = []
    # list of mus notes
    mus = []

    # men
    musPreferences = {}
    # women
    recPreferences = {}
    parsingMatches = True

    songNames = []

    # see java code for input format
    print('starting parse on ' + filename)
    i = 0
    while i < len(lines):
        line = lines[i]
        #print(line)
        # starting to read data from new song
        if line == '---\n' or line == '---\r\n':
            musList.append(mus)
            recList.append(rec)
            print('new song')
            songCount += 1
            print('matching')
            matchesMap = performMatching(mus, rec, musPreferences, recPreferences)
            print('done matching')
            matchesMapList.append(matchesMap)
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
        elif line == '***\n' or line == '***\r\n':
            parsingMatches = False
            if i + 1 < len(lines):
                name = lines[i+1]
                songNames.append(name)
            i += 1
        # continuing to read matching candidate data
        elif parsingMatches:
            arr = line.split(":")
            musNote = decodeNote(arr[0][1:-1])
            musIndex = musNote['index']
            mus.append(musNote)
            musPreferences[musIndex] = PriorityQueue()
            #print(arr)
            if arr[1] == '\n' or arr[1] == '\r\n':
                print('skipped')
                i+=1
                continue
            arr = arr[1].split(';')
            for str in arr:
                recNote, lcs, percent, distance = str.split('/')
                recNote = decodeNote(recNote[1:-1])
                recIndex = recNote['index']
                lcs = float(lcs)
                percent = float(percent)
                distance = float(distance)
                # calculate cost. smaller cost is better
                cost = 0.1 * ((1 - lcs) + (1 - percent)) + 0.9 * distance
                musPreferences[musIndex].update(recIndex, cost)
                if recIndex not in recPreferences:
                    recPreferences[recIndex] = PriorityQueue()
                recPreferences[recIndex].update(musIndex, cost)
        # parsing rec notes
        elif not parsingMatches:
            recNote = decodeNote(line)
            rec.append(recNote)
        else:
            assert (False)
        i+=1
    print('ending parse')
    assert(len(musList) == len(recList))
    return musList, recList, matchesMapList, songNames

def decodeNote(str):
    end = str.find("}")
    if end == -1:
        end = len(str)
    str = str[str.find("{") + 1 : end]
    note = {}
    arr = str.split(',')
    note['key'] = int(arr[0])
    note['index'] = int(arr[1])
    note['onv'] = int(arr[2])
    note['offv'] = int(arr[3])
    note['start'] = int(arr[4])
    note['end'] = int(arr[5])
    if note['end'] < 0 or note['start'] < 0:
        print(arr)
    note['track'] = int(arr[6])
    return note
	
def printNote(note):
    return "{{{},{},{},{},{},{},{}}}".format(note['key'], note['index'],note['onv'],note['offv'],
        note['start'],note['end'],note['track'])



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

    #with open("matchedNotes.txt", 'w') as of:
    #    for pair in matches:
    #        of.write("{},{}\n".format(pair[0],pair[1]))

    return matchesMap

UNMATCHED_PERCENT = 0
def findMatches(musLen, recLen, musPreferences, recPreferences):
    #list of (mus, rec)
    matches = []
    musFree = [i for i in range(musLen)]
    # a certain proportion will remain unmatched
    endLength = len(musFree) * UNMATCHED_PERCENT
    recFree = [i for i in range(recLen)]
    while len(musFree) > int(endLength):
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

############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return whether |state| is an end state or not.
    def isEnd(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    # First, call solve on the desired SearchProblem |problem|.
    # Then it should set two things:
    # - self.actions: list of actions that takes one from the start state to an end
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem): raise NotImplementedError("Override me")

############################################################
# Uniform cost search algorithm (Dijkstra's algorithm).

class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def solve(self, problem):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.startState()
        frontier.update(startState, 0)

        prevIndex = -1

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, pastCost = frontier.removeMin()
            if state == None: break
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print("Exploring %s with pastCost %s" % (state, pastCost))

            # Check if we've reached an end state; if so, extract solution.
            if problem.isEnd(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print("numStatesExplored = %d" % self.numStatesExplored)
                    print("totalCost = %s" % self.totalCost)
                    print("actions = %s" % self.actions)
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.succAndCost(state, prevIndex):
                if self.verbose >= 3:
                    print("  Action %s => %s with cost %s + %s" % (action, newState, pastCost, cost))
                if frontier.update(newState, pastCost + cost):
                    # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)
            prevIndex = state[0]
        if self.verbose >= 1:
            print("No path found")

# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left...

############################################################
# Simple examples of search problems to test your code for Problem 1.

# A simple search problem on the number line:
# Start at 0, want to go to 10, costs 1 to move left, 2 to move right.
class NumberLineSearchProblem:
    def startState(self): return 0
    def isEnd(self, state): return state == 10
    def succAndCost(self, state): return [('West', state-1, 1), ('East', state+1, 2)]

# A simple search problem on a square grid:
# Start at init position, want to go to (0, 0)
# cost 2 to move up/left, 1 to move down/right
class GridSearchProblem(SearchProblem):
    def __init__(self, size, x, y): self.size, self.start = size, (x,y)
    def startState(self): return self.start
    def isEnd(self, state): return state == (0, 0)
    def succAndCost(self, state):
        x, y = state
        results = []
        if x-1 >= 0: results.append(('North', (x-1, y), 2))
        if x+1 < self.size: results.append(('South', (x+1, y), 1))
        if y-1 >= 0: results.append(('West', (x, y-1), 2))
        if y+1 < self.size: results.append(('East', (x, y+1), 1))
        return results