from src import util
import matplotlib.pyplot as plt

def main():
    n_files = 37
    musList, recList, predictedMatchesMapList, songNames = util.match_by_predict(n_files=n_files)

    matchesMapList = predictedMatchesMapList
    #recList, matchesMapList = util.trim(recList, matchesMapList)
    for i in range(len(recList)):
        rec = recList[i]
        matched = []
        actual = []
        for musIndex in sorted(matchesMapList[i].keys()):
            note = rec[matchesMapList[i][musIndex]]
            matched.append(note['start_normal'])
        for note in recList[i]:
            actual.append(note['start_normal'])
        plt.plot(matched)
        plt.plot(actual)
        plt.title(songNames[i])
        plt.show()


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()