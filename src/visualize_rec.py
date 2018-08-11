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
import argparse
import pickle

def main(args):
    n_files = 33
    #musList, recList, predictedMatchesMapList, songNames = util.match_by_predict(n_files=n_files)
    if args.files is None:
        args.files = list(range(n_files))
    print(args.files)
    for file_index in args.files:
        if os.path.exists('matchings/match{}'.format(file_index)):
            print('file exists')
            exit()
        if os.path.exists('plots/plot{}.png'.format(file_index)):
            print('file exists')
            exit()

    musList, musNames, recList, recNames = src.rnn_util.loadSongLists(files=args.files, for_predict=False)
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
        pylab.savefig('plots/plot{}.png'.format(args.files[i]))
        try:
            with open('matchings/match{}'.format(args.files[i]), 'wb') as pickle_file:
                pickle.dump(matches, pickle_file)
            print('Save matching succeeded')
        except:
            print('Failed to save matching')
        pylab.clf()


if __name__ == '__main__':
    ### TODO add support for setting the model from the command line

    parser = argparse.ArgumentParser(description='Trains and tests the model.')
    subparsers = parser.add_subparsers()

    parser.add_argument('-f', '--files', nargs='*', help='Set flag', required=False)


    ARGS = parser.parse_args()
    main(ARGS)