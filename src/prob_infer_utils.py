import random
from src import util
import statistics
import copy

TIMESTEPS = 70
n_epochs = 50
n_samples = 1
epsilon = 0.001
relativeNoteDurations = [2, 1+3/4, 1+1/2, 1+1/4, 1,
              1/2+1/4, 1/2, 1/4+1/8, 1/8+1/16, 1/8,
              (1/4)*(1/3), 1/16, (1/4)*(1/6)]
relativeNoteDurations = [d * 4 for d in relativeNoteDurations]
songSpecificNoteDurations = []

durationPatterns = {}

def inferMus(rec, variables):
    mus = copy.deepcopy(rec)
    curStart = 0
    for i in range(len(mus)):
        mus[i]['start'] = curStart
        mus[i]['end'] = curStart + variables[i].duration
        if i != len(mus) - 1 and variables[i+1].chord == 0:
            curStart += variables[i].duration
    musList, recList = util.normalizeIndices([mus], [rec])
    return musList[0]


def preprocess(rec):
    global durationPatterns
    global songSpecificNoteDurations
    durationPatterns = {i: [] for i in range(len(rec))}
    noteDurations = []
    for i in range(len(rec)):
        noteDurations.append(rec[i]['end'] - rec[i]['start'])
        start = util.tsloop_start(i, timesteps=TIMESTEPS)
        end = util.tsloop_end(i, len(rec), timesteps=TIMESTEPS)
        rhythmList, centerIndex = notesAsRhythmList(i - start, notes=rec[start:end+1])
        lh = centerIndex
        rh = centerIndex
        durationPatterns[i].append([rhythmList[centerIndex]])
        while True:
            if lh < start and rh > end:
                break
            if lh >= start:
                lh -= 1
                durationPatterns[i].append(rhythmList[lh:rh + 1])
            if rh <= end:
                rh += 1
                durationPatterns[i].append(rhythmList[lh:rh + 1])
    quarterNoteLen = statistics.median(noteDurations)
    print(quarterNoteLen)
    songSpecificNoteDurations = [r * quarterNoteLen for r in relativeNoteDurations]

def notesAsRhythmList(index, notes=None, variables=None):
    rhythms = []
    i = 0
    indexInRhythmList = index
    if notes is None:
        length = len(variables)
    else:
        length = len(notes)
    alreadyChanged = False
    while i < length:
        if notes is None:
            rhythms.append(variables[i].duration)
        else:
            rhythms.append(notes[i]['end'] - notes[i]['start'])
        i = nextChord(i, notes=notes, variables=variables)
        if i > index and not alreadyChanged:
            indexInRhythmList = len(rhythms) - 1
            alreadyChanged = True
    return rhythms, indexInRhythmList


def computeWeight(i, variables, note, assignment, durationPatterns):
    print("Trying {}, {}".format(assignment.duration, assignment.chord))
    noteLength = note['end'] - note['start']
    assert(noteLength >= 0)
    durationWeight = 1.0 / (max(abs(noteLength - assignment.duration), 1) + epsilon)

    rhythmList, centerIndex = notesAsRhythmList(i, variables=variables)
    print(rhythmList)
    start = i + util.tsloop_start(i, timesteps=TIMESTEPS)
    end = i + util.tsloop_end(i, len(variables), timesteps=TIMESTEPS)
    lh = centerIndex
    rh = centerIndex
    maxSize = 1
    while True:
        if lh < start and rh > end:
            break
        if lh >= start:
            lh -= 1
            if rhythmList[lh:rh + 1] in durationPatterns[i]:
                maxSize += 1
        if rh <= end:
            rh += 1
            if rhythmList[lh:rh + 1] in durationPatterns[i]:
                maxSize += 1
    patternWeight = maxSize / len(rhythmList)
    print(rhythmList[lh:rh + 1])
    print(durationPatterns[i])
    print(patternWeight)
    print(durationWeight)
    print('\n')
    return 0.5 * patternWeight + 0.5 * durationWeight

def findMaxWeight(var_list, rec, durationPatterns):
    bestIndex = 0
    bestWeight = float('-inf')
    for v in range(len(var_list)):
        weight = 0
        for j in range(len(var_list[v])):
            weight += computeWeight(j, var_list[v], rec[j], var_list[v][j], durationPatterns)
        if weight > bestWeight:
            bestWeight = weight
            bestIndex = v
    return var_list[bestIndex]

def nextChord(i, notes=None, variables=None):
    if notes is None:
        # if you start me on the head of a chord, i'll assume you want to find the next one
        if variables[i].chord == 0:
            i += 1
        while i < len(variables) and variables[i].chord == 1:
            i += 1
        return i
    else:
        curStart = notes[i]['start']
        while i < len(notes) and notes[i]['start'] == curStart:
            i += 1
        return i


class Var():
    # duration
    # chord is 0 if head of chord or singular, 1 if part of preceding note's chord
    def __init__(self, songSpecificNoteDurations, duration=None, chord=None):
        if duration is None or chord is None:
            self.duration = random.choice(songSpecificNoteDurations)
            self.chord = random.choice([0, 1])
        else:
            self.duration = duration
            self.chord = chord

    def __repr__(self):
        str = "Var({}, ".format(self.duration)
        if self.chord:
            str += "chord)"
        else:
            str += "head)"
        return str
