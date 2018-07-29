import sys
import os
from pathlib import Path
import pylab
cwd = Path(os.getcwd())
sys.path.append(cwd.parent.__str__())
sys.path.append(os.getcwd())
print(sys.path)
import src
import src.rnn_util
import src.matching_util
import src.note_util

def main():
    n_files = 37
    #musList, recList, predictedMatchesMapList, songNames = util.match_by_predict(n_files=n_files)
    musList, musNames, recList, recNames = src.rnn_util.loadSongLists(for_predict=False)
    musList, recList = src.note_util.normalizeTimes(musList, recList)
    musList, recList = src.note_util.normalizeIndices(musList, recList)
    #recList, matchesMapList = util.trim(recList, matchesMapList)
    assert(len(musList) == len(recList))
    for i in range(len(recList)):
        rec = recList[i]
        mus = musList[i]
        matches = src.matching_util.get_matching(mus, rec)
        matched = []
        actual = []
        for musIndex in sorted(matches.keys()):
            note = rec[matches[musIndex]]
            matched.append(note['start_normal'])
        for note in recList[i]:
            actual.append(note['start_normal'])
        pylab.plot(matched, label='matched')
        pylab.plot(actual, label='actual rec')
        pylab.title(musNames[i])
        pylab.legend(loc='upper left')
        pylab.show()


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()