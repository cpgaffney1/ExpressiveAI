import heapq, collections, re, sys, time, os, random, time
import numpy as np
import copy
from keras.models import load_model

#########################################################
# util functions for music

#note that midi has 128 notes, more than the 88 on a keyboard
KEYBOARD = 128
TIMESTEPS = 20
N_FILES = 37
N_MATCHING_FEATURES = 4
N_ML_FEATURES = 8

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

def dataAsWindowTwoSided(musList, recList, matchesMapList):
    print('transforming')
    songLengths = [len(mus.keys()) for mus in matchesMapList]
    numObs = sum(songLengths) - len(matchesMapList) * (2 * TIMESTEPS + 1) - len(matchesMapList)
    print(numObs)
    dim = (2 * TIMESTEPS + 1) * 6
    x = np.zeros((numObs, dim))
    y = np.zeros((numObs, 2))
    obs_index = 0
    for song_num in range(len(musList)):
        print(song_num)
        # list of mus_start, mus length, key, onv, rec offset, rec len offset for each timestep
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
                x_obs.append(mus[musIndex]['start_normal'])
                assert(mus[musIndex]['end_normal'] - mus[musIndex]['start_normal'] >= 0)
                x_obs.append(mus[musIndex]['end_normal'] - mus[musIndex]['start_normal'])
                x_obs.append(mus[musIndex]['key'])
                x_obs.append(mus[musIndex]['onv'])
                x_obs.append(rec[prevRecIndex]['offset'])
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

def splitDataTwoSided(x, only_mus=False):
    if only_mus:
        n_step_features = 4
    else:
        n_step_features = 6
    x_train = np.zeros((len(x), 2 * TIMESTEPS + 1, n_step_features))
    for i in range(len(x)):
        x_obs = np.zeros((2 * TIMESTEPS + 1, n_step_features))
        for t in range((x_obs.shape[0])):
            for j in range((x_obs.shape[1])):
                offset = (t * 6)
                x_obs[t][j] = x[i][offset + j]
        x_train[i] = x_obs
    return x_train

def assemble_matching_data_for(songIndex, musList, recList, potentialMatchesMapList, matchesMapList, for_predict, is_ml):
    s = potentialMatchesMapList[songIndex]
    obsCount = 0
    obsCount += len(s.keys())
    for k in s.keys():
        obsCount += len(s[k])

    x = np.zeros((obsCount, TIMESTEPS * 2 + 1, N_MATCHING_FEATURES * 2))
    y = np.zeros(obsCount)

    obs_index = 0
    print("starting song at index {}".format(songIndex))
    position = -1
    matchesKeys = sorted(matchesMapList[songIndex].keys())
    for i in range(len(matchesKeys)):
        musIndex = matchesKeys[i]
        position += 1
        if position < TIMESTEPS or position >= len(matchesKeys) - TIMESTEPS:
            continue

        ### add the actual predicted match
        #print(matchesMapList[songIndex][musIndex])
        #print(potentialMatchesMapList[songIndex][musIndex])
        assert (matchesMapList[songIndex][musIndex] in potentialMatchesMapList[songIndex][musIndex])
        matches = potentialMatchesMapList[songIndex][musIndex]
        recIndex = matchesMapList[songIndex][musIndex]
        ### important to remove the actual match
        matches.remove(recIndex)
        if len(matches) == 0:
            continue

        observation = indiv_matching_obs(musList, recList, musIndex, recIndex, songIndex)
        x[obs_index] = observation
        y[obs_index] = 1
        obs_index += 1

        #### here depends on whether predicting or not
        if for_predict:
            for potential_rec_match in matches:
                recIndex = potential_rec_match
                observation = indiv_matching_obs(musList, recList, musIndex, recIndex, songIndex)
                x[obs_index] = observation
                y[obs_index] = 0
                obs_index += 1
        else:
            recIndex = random.choice(matches)
            observation = indiv_matching_obs(musList, recList, musIndex, recIndex, songIndex)
            x[obs_index] = observation
            y[obs_index] = 0
            obs_index += 1

    deleteIndexes = []
    for i in range(x.shape[0]):
        assert (x[i].shape == np.zeros((TIMESTEPS * 2 + 1, N_MATCHING_FEATURES * 2)).shape)
        if np.array_equal(x[i], np.zeros((TIMESTEPS * 2 + 1, N_MATCHING_FEATURES * 2))):
            deleteIndexes.append(i)
    x = np.delete(x, deleteIndexes, 0)
    y = np.delete(y, deleteIndexes)


    if is_ml:
        x_raw = x
        x = construct_ml_data(x_raw)
        return x_raw, x, y, x.shape[0]
    else:
        return x, x, y, x.shape[0]

def indiv_matching_obs(musList, recList, musIndex, recIndex, songIndex):
    # musIndex is said to be paired with recIndex. We are trying to predict whether this is the case or not
    observation = np.zeros((TIMESTEPS * 2 + 1, N_MATCHING_FEATURES * 2))
    for i in range(-TIMESTEPS, TIMESTEPS + 1):
        if musIndex + i < 0 or musIndex + i >= len(musList[songIndex]):
            continue
        if recIndex + i < 0 or recIndex + i >= len(recList[songIndex]):
            continue
        musNote = musList[songIndex][musIndex + i]
        recNote = recList[songIndex][recIndex + i]
        step = np.zeros(N_MATCHING_FEATURES * 2)
        # features of step, 4 for mus and 4 for matching rec note (or not matching)
        # order start normal, end normal, key, onv, index
        step[0] = musNote['start_normal']
        step[1] = musNote['end_normal']
        step[2] = musNote['key']
        step[3] = musNote['index_normal']
        # corresponding rec note
        step[4] = recNote['start_normal']
        step[5] = recNote['end_normal']
        step[6] = recNote['key']
        step[7] = recNote['index_normal']
        if i == 0:
            assert (step[2] == step[6])
        observation[i + TIMESTEPS] = step
    return observation


def match_by_predict(n_files):
    is_ml = False
    musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList = parseMatchedInput(
        'C://Users//cpgaf//PycharmProjects//ExpressiveAI//javaOutput/javaOutput', range(0, n_files))
    musList, recList = normalizeTimes(musList, recList)
    musList, recList = normalizeIndices(musList, recList)
    predictor_model = load_model("C://Users//cpgaf//PycharmProjects//ExpressiveAI//src//matching_model.h5")

    predictedMatchesMapList = []

    for songIndex in range(len(potentialMatchesMapList)):
        predictedMatchesMapList.append({})
        for_predict = True
        x_raw, x, y, _ = assemble_matching_data_for(songIndex, musList, recList, potentialMatchesMapList,
                                                            matchesMapList, for_predict, is_ml)
        predictions = predictor_model.predict(x)
        print(predictions)
        print(np.count_nonzero(np.ndarray.astype(np.round(predictions[:, 1]), np.int)))
        print('\n')

        ####
        ### perform prediction for current song
        used_rec_indices = []
        lastMusIndex = musList[songIndex][-1]['index']
        lastRecIndex = recList[songIndex][-1]['index']
        prevMusIndex = denormalizeIndex(x_raw[0][TIMESTEPS][3], lastMusIndex)
        indices = []
        for i in range(x.shape[0]):
            curMusIndex = denormalizeIndex(x_raw[i][TIMESTEPS][3], lastMusIndex)

            if curMusIndex > prevMusIndex:
                # moved to predictions for next mus index
                if is_ml:
                    prediction = predict_matches_for_indices_ml(x, x_raw, predictor_model, indices, used_rec_indices,
                                                         lastMusIndex, lastRecIndex)
                else:
                    prediction = predict_matches_for_indices_nn(x, predictor_model, indices, used_rec_indices,
                                                                lastMusIndex, lastRecIndex)
                if np.isnan(prediction[1]):
                    #predictedMatchesMapList.pop(prediction[0], None)
                    pass
                else:
                    assert (prediction[0] == int(prediction[0]))
                    assert(prediction[1] == int(prediction[1]))
                    predictedMatchesMapList[songIndex][prediction[0]] = prediction[1]
                    used_rec_indices.append(prediction[1])
                    assert(predictedMatchesMapList[songIndex][prediction[0]] == int(predictedMatchesMapList[songIndex][prediction[0]]))
                indices = []
            # else:
            # curMusIndex == prevMusIndex
            # still collecting predictions for a given index
            prevMusIndex = curMusIndex
            indices.append(i)
        for musNote in matchesMapList[songIndex]:
            if musNote not in predictedMatchesMapList[songIndex]:
                predictedMatchesMapList[songIndex][musNote] = matchesMapList[songIndex][musNote]
    return musList, recList, predictedMatchesMapList, songNames

def predict_matches_for_indices_nn(x, predictor_model, indices, used_rec_indices,
                                lastMusIndex, lastRecIndex):
    predictions = []
    musIndex = denormalizeIndex(x[indices[0]][TIMESTEPS][3], lastMusIndex)
    #print('\n')
    for i in indices:
        #print('mus: {}, rec: {}'.format(musIndex,
        #                                denormalizeIndex(x_raw[i][TIMESTEPS][7], lastRecIndex)))
        temp = np.zeros((1, x[i].shape[0], x[i].shape[1]))
        temp[0] = x[i]
        y = predictor_model.predict_on_batch([temp])[0]
        #print(y)
        y = y[1]
        predictions.append(y)
    argmax = np.nanargmax(predictions)
    recIndex = denormalizeIndex(x[indices[argmax]][TIMESTEPS][7], lastRecIndex)
    #print(predictions)
    return (musIndex, recIndex)

def predict_matches_for_indices_ml(x, x_raw, predictor_model, indices, used_rec_indices,
                                lastMusIndex, lastRecIndex):
    predictions = []
    musIndex = denormalizeIndex(x_raw[indices[0]][TIMESTEPS][3], lastMusIndex)
    #print('\n')
    for i in indices:
        #print('mus: {}, rec: {}'.format(musIndex,
        #                                denormalizeIndex(x_raw[i][TIMESTEPS][7], lastRecIndex)))
        temp = np.zeros((1, x[i].shape[0]))
        temp[0] = x[i]
        y = predictor_model.predict_on_batch([temp])[0]
        #print(y)
        y = y[1]
        predictions.append(y)
    argmax = np.nanargmax(predictions)
    recIndex = denormalizeIndex(x_raw[indices[argmax]][TIMESTEPS][7], lastRecIndex)
    #print(predictions)
    return (musIndex, recIndex)

def load_data(core_input_shape=5, n_files=N_FILES):
    musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList = parseMatchedInput('javaOutput/javaOutput', range(0,n_files))
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


def weight_experiment(n_files = N_FILES):
    bestWeights = (0,0,0,0)
    weights = (0.008838140076199442, 0.00038917980183811134, 0.9402576634560431, 0.25051501666591885)
    s = sum(weights)
    bestValue = float('inf')
    print("beginning trial")
    n_epochs = 1000
    for t in range(n_epochs):
        weights = tuple([w * s / sum(weights) for w in weights])
        start = time.time()
        print("Trying weight: {}".format(weights))
        musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList = parseMatchedInput('javaOutput/javaOutput', range(0, n_files), activeWeights=weights)
        if matchValue < bestValue:
            bestValue = matchValue
            bestWeights = weights
        print("time: {}".format(time.time() - start))
        print("value: {}".format(matchValue))
        print("current best value: {}, best: {}".format(bestValue, bestWeights))
        weights = (weights[0] + random.uniform(-1,1)*0.0001, weights[1] + random.uniform(-1,1)*0.0001,
                   weights[2] + random.uniform(-1,1)*0.01, weights[3] + random.uniform(-1,1)*0.01)
    print("Min weights:")
    print(bestWeights)
    exit()

def load_data_rnn(n_files=N_FILES):
    #enhanced_nn_model = load_model("enhanced_nn_model.h5")
    #prelimSongPredictions = enhanced_nn_predict.predict(enhanced_nn_model, fromFile='javaOutput/javaOutput', files=range(0,n_files))
    musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList = parseMatchedInput('javaOutput/javaOutput', range(0,n_files))
    musList, recList = normalizeTimes(musList, recList)
    recList, matchesMapList = trim(recList, matchesMapList)
    recList = addOffsets(musList, recList, matchesMapList)
    x, y = dataAsWindowTwoSided(musList, recList, matchesMapList)
    x_train = x.astype('float32')
    y_train = y.astype('float32')
    x_train = splitDataTwoSided(x_train, only_mus=True)
    x_train = x_train.astype('float32')
    return x_train, y_train



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

def normalizeIndices(musList, recList):
    for i in range(len(musList)):
        lastIndex = float(musList[i][-1]['index'])
        for note in musList[i]:
            note['index_normal'] = (note['index'] * 100) / lastIndex
    for i in range(len(recList)):
        lastIndex = float(recList[i][-1]['index'])
        for note in recList[i]:
            note['index_normal'] = (note['index'] * 100) / lastIndex
    return musList, recList

def denormalizeIndices(indices, lastIndex):
    for i in range(len(indices)):
        indices[i] *= (lastIndex / 100.0)
        indices[i] = int(indices[i])
    return indices

def denormalizeIndex(index, lastIndex):
    return int(index * lastIndex / 100.0)

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

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def construct_ml_data(x_raw):
    x_train = np.zeros((x_raw.shape[0], N_ML_FEATURES))
    for i in range(x_raw.shape[0]):
        if i % 500 == 0:
            print('progress: {}%'.format(int(float(i) / x_raw.shape[0] * 100)))
        # average absolute start time norm difference
        x_train[i][0] = np.linalg.norm(np.abs(x_raw[i,:,0] - x_raw[i,:,4]))
        # average absolute end time norm difference
        x_train[i][1] = np.linalg.norm(np.abs(x_raw[i, :, 1] - x_raw[i, :, 5]))
        # average absolute normalized index time difference
        x_train[i][2] = np.linalg.norm(np.abs(x_raw[i, :, 4] - x_raw[i, :, 7]))
        # norm of vector difference in notes
        x_train[i][3] = np.linalg.norm(x_raw[i,:,2] - x_raw[i,:,6])
        # edit distance
        x_train[i][4] = levenshteinDistance(''.join(np.ndarray.astype(x_raw[i,:,2], np.str)),
                                            ''.join(np.ndarray.astype(x_raw[i,:,6], np.str)))
        # start comparison
        x_train[i][5] = np.abs(x_raw[i][TIMESTEPS][0] - x_raw[i][TIMESTEPS][4])
        # end comparison
        x_train[i][6] = np.abs(x_raw[i][TIMESTEPS][1] - x_raw[i][TIMESTEPS][5])
        # index comparison
        x_train[i][7] = np.abs(x_raw[i][TIMESTEPS][3] - x_raw[i][TIMESTEPS][7])
    return x_train

def parseMatchedInput(filename, fileNums, exclude=[], activeWeights=(0.008838140076199442, 0.00038917980183811134, 0.9402576634560431, 0.25051501666591885)):
    musList = []
    recList = []
    matchesMapList = []
    potentialMatchesMapList = []
    songNames = []
    matchValue = 0
    for i in fileNums:
        if i in exclude:
            continue
        ml, rl, mml, sn, mv, pmml = parseMatchedInputFromFile(filename + str(i) + '.txt', activeWeights)
        musList += ml
        recList += rl
        matchesMapList += mml
        potentialMatchesMapList += pmml
        songNames += sn
        matchValue += mv
    return musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList


def parseMatchedInputFromFile(filename, activeWeights):
    file = filename
    with open(file) as f:
        lines = f.readlines()
    songCount = 0

    # initialize storage variables
    matchesMapList = []
    potentialMatchesMapList = []
    musList = []
    recList = []
    # list of rec notes
    rec = []
    # list of mus notes
    mus = []

    potentialMatchesMap = {}

    # men
    musPreferences = {}
    # women
    recPreferences = {}
    parsingMatches = True

    songNames = []
    matchValue = 0

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
            matchesMap, matchValue = performMatching(mus, rec, musPreferences, recPreferences)
            print('done matching')
            matchesMapList.append(matchesMap)
            potentialMatchesMapList.append(potentialMatchesMap)
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

            # list of all potential matches
            potentialMatchesMap[musIndex] = []

            mus.append(musNote)
            musPreferences[musIndex] = PriorityQueue()
            #print(arr)
            if arr[1] == '\n' or arr[1] == '\r\n':
                print('skipped')
                i+=1
                continue
            arr = arr[1].split(';')
            for str in arr:
                recNote, lcs, percent, distance, editDist = str.split('/')
                recNote = decodeNote(recNote[1:-1])
                recIndex = recNote['index']

                # add rec index to potential match list for this musIndex
                potentialMatchesMap[musIndex].append(recIndex)

                lcs = float(lcs)
                percent = float(percent)
                distance = float(distance)
                editDist = float(editDist)
                # calculate cost. smaller cost is better
                #######################################################
                cost = activeWeights[0]*(1/(lcs+0.0001)) + activeWeights[1]*(1/(percent+0.0001)) + activeWeights[2]*distance + activeWeights[3]*editDist
                assert(cost >= 0)
                #######################################################
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
    return musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList

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
    return "{{{},{},{},{},{},{}}}".format(note['key'], note['index'],note['onv'],note['offv'],
        note['start'],note['end'],note['track'])



def performMatching(mus, rec, musPreferences, recPreferences):
    musPreferencesOrdered = {}
    for i in musPreferences:
        pq = musPreferences[i]
        pref = copy.deepcopy(pq.priorities)
        lis = []
        while True:
            min = pq.removeMin()
            if min == (None, None): break
            lis.append(min[0])
        musPreferencesOrdered[i] = lis
        musPreferences[i] = PriorityQueue()
        for key in pref.keys():
            musPreferences[i].update(key, pref[key])

    recPreferencesOrdered = {}
    for i in recPreferences:
        pq = recPreferences[i]
        pref = copy.deepcopy(pq.priorities)
        lis = []
        while True:
            min = pq.removeMin()
            if min == (None, None): break
            lis.append(min[0])
        recPreferencesOrdered[i] = lis
        recPreferences[i] = PriorityQueue()
        for key in pref.keys():
            recPreferences[i].update(key, pref[key])

    matchesMap = {}
    matches, matchValue = findMatches(len(mus), len(rec), musPreferences, recPreferences,
                                      musPreferencesOrdered, recPreferencesOrdered)
    for m in matches:
        matchesMap[m[0]] = m[1]
    #print(matchesMap)
    #print('  ' + ',\n  '.join('Mus %s is matched to rec %s' % m for m in matches))

    #with open("matchedNotes.txt", 'w') as of:
    #    for pair in matches:
    #        of.write("{},{}\n".format(pair[0],pair[1]))

    return matchesMap, matchValue

UNMATCHED_PERCENT = 0
def findMatches(musLen, recLen, musPreferences, recPreferences,
                musPreferencesOrdered, recPreferencesOrdered):
    matchValue = 0
    #list of (mus, rec)
    matches = []
    musFree = [i for i in range(musLen)]
    # a certain proportion will remain unmatched
    endLength = len(musFree) * UNMATCHED_PERCENT
    recFree = [i for i in range(recLen)]
    while len(musFree) > int(endLength):
        currMus = musFree[0]
        # all his choices are gone, so sad. He gets no match
        if len(musPreferencesOrdered[currMus]) == 0:
            musFree.pop(0)
            continue
        recChoice = musPreferencesOrdered[currMus][0]
        if recChoice in recFree:
            matchValue += musPreferences[currMus].priorities[recChoice] + recPreferences[recChoice].priorities[currMus]
            matches.append((currMus, recChoice))
            musFree.pop(0)
            recFree.remove(recChoice)
        else:
            currMatch = findRecPair(matches, recChoice)
            if currMatch == None:
                print(recChoice)
            # rec would prefer currMus over its current partner
            if recPreferencesOrdered[recChoice].index(currMus) < recPreferencesOrdered[recChoice].index(currMatch[0]):
                matchValue = matchValue - recPreferences[recChoice].priorities[currMatch[0]] - \
                    musPreferences[currMatch[0]].priorities[recChoice]
                matchValue = matchValue + recPreferences[recChoice].priorities[currMus] + \
                    musPreferences[currMus].priorities[recChoice]
                musFree.pop(0)
                musFree.append(currMatch[0])
                matches.remove(currMatch)
                matches.append((currMus, recChoice))
        musPreferencesOrdered[currMus].pop(0)
    return matches, matchValue

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