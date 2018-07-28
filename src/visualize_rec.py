import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
cwd = Path(os.getcwd())
sys.path.append(cwd.parent)
sys.path.append(os.getcwd())
import src
import src.rnn_util
import src.matching_util

def main():
    n_files = 37
    #musList, recList, predictedMatchesMapList, songNames = util.match_by_predict(n_files=n_files)
    musList, musNames, recList, recNames = src.rnn_util.loadSongLists(for_predict=False)
    #recList, matchesMapList = util.trim(recList, matchesMapList)
    assert(len(musList) == len(recList))
    for i in range(len(recList)):
        rec = recList[i]
        mus = musList[i]
        matches = src.matching_util.get_matching(mus, rec)
        print(matches)
        exit()
        matched = []
        actual = []
        for musIndex in sorted(matches.keys()):
            note = rec[matches[musIndex]]
            matched.append(note['start_normal'])
        for note in recList[i]:
            actual.append(note['start_normal'])
        plt.plot(matched)
        plt.plot(actual)
        plt.title(musNames[i])
        plt.show()


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()