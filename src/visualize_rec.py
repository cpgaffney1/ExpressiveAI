import sys
import os
from pathlib import Path
import pylab
cwd = Path(os.getcwd())
sys.path.append(cwd.parent.__str__())
sys.path.append(os.getcwd())
print(sys.path)
import src
import src.util.rnn_util
import src.matching_util
import src.util.note_util
import argparse
import pickle

n_files = 56


def match(args):
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

    musList, musNames, recList, recNames = src.util.rnn_util.loadSongLists(files=args.files, for_predict=False)
    musList, recList = src.util.note_util.normalizeTimes(musList, recList)
    musList, recList = src.util.note_util.normalizeIndices(musList, recList)
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

def load_and_vis(args):
    index_to_match_map = src.util.rnn_util.load_match_maps()
    print(len(index_to_match_map))
    musList, musNames, recList, recNames = src.util.rnn_util.loadSongLists(files=list(range(n_files)), for_predict=False)
    for i in range(len(recList)):
        print(i)
        rec = recList[i]
        mus = musList[i]
        match = index_to_match_map.get(str(i), None)
        if match is None:
            continue
        x = []
        y = []
        for mus_index in range(len(mus)):
            rec_index = match.get(mus_index, None)
            x.append(mus[mus_index]['start'])
            if rec_index is None:
                y.append(0)
            else:
                assert(mus[mus_index]['key'] == rec[rec_index]['key'])
                y.append(rec[rec_index]['start'])

        pylab.plot(x, y)
        pylab.title(musNames[i])
        pylab.show()


if __name__ == '__main__':
    ### TODO add support for setting the model from the command line

    parser = argparse.ArgumentParser(description='Trains and tests the model.')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('match', help='')
    command_parser.add_argument('-f', '--files', nargs='*', help='Set flag', required=False)
    command_parser.set_defaults(func=match)


    command_parser = subparsers.add_parser('load_and_vis')
    command_parser.set_defaults(func=load_and_vis)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        exit(1)
    else:
        ARGS.func(ARGS)