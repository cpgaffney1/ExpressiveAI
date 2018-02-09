from src import util
from src.prob_infer_utils import Var
from src import prob_infer_utils as p_util
import sys
from multiprocessing import Pool

def main():
    n_files = 1
    musList, recList, matchesMapList, songNames, matchValue, potentialMatchesMapList = util.parseMatchedInput(
        '../javaOutput/javaOutput', range(0, n_files))
    musList, recList = util.normalizeTimes(musList, recList)

    print(p_util.relativeNoteDurations)
    for s in range(1):
        rec = recList[s]
        p_util.preprocess(rec)
        name = songNames[s]
        print('\n')
        print(name)
        pool = Pool(3)

        var_results = []
        for i in range(p_util.n_samples):
            print('processing sample {}'.format(i))
            var_results.append(pool.apply_async(runUntilConvergence,
                                                (rec, p_util.songSpecificNoteDurations, p_util.durationPatterns)))

        var_samples = []
        for result in var_results:
            var_samples.append(result.get())

        bestAssign = p_util.findMaxWeight(var_samples, rec, p_util.durationPatterns)
        mus = p_util.inferMus(rec, bestAssign)
        file = "C://Users//cpgaf//PycharmProjects//ExpressiveAI//inferredMus//inferred" + str(s) + ".txt"
        with open(file, 'w') as of:
            for note in mus:
                of.write(util.printNote(note) + '\n')
        p_util.durationPatterns = {}


def runUntilConvergence(rec, songSpecificNoteDurations, durationPatterns):
    variables = [Var(songSpecificNoteDurations) for _ in rec]
    variables[0].chord = 0
    for t in range(p_util.n_epochs):
        print('epochs {}% done'.format(int(float(t) / p_util.n_epochs * 100)))
        print(variables)
        for j in range(len(variables)):
            bestVar = bestAssignment(j, variables, rec, songSpecificNoteDurations, durationPatterns)
            variables[j] = bestVar
    return variables

def bestAssignment(j, variables, rec, songSpecificNoteDurations, durationPatterns):
    bestVar = None
    bestWeight = float('-inf')
    noteLength = rec[j]['end'] - rec[j]['start']
    closestIndex = min(range(len(songSpecificNoteDurations)), key=lambda i: abs(songSpecificNoteDurations[i] - noteLength))
    sideRange = 2
    for d in songSpecificNoteDurations[max(0,closestIndex-sideRange):min(closestIndex+sideRange,len(songSpecificNoteDurations))]:
        for ic in [0, 1]:
            var = Var(songSpecificNoteDurations, duration=d, chord=ic)
            w = p_util.computeWeight(j, variables, rec[j], var, durationPatterns)
            if w > bestWeight:
                bestWeight = w
                bestVar = var
    return bestVar


if __name__ == '__main__':
    main()

