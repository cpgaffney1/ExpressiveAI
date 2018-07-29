import numpy as np
import copy
import time

N_BRANCHES = 100
MAX_SKIP_FRACTION = 0.05
n_max_skip = 0
potential_matches = []
reached = False

def get_matching(mus, rec):
    print('beginning matching')
    global potential_matches
    potential_matches = []
    global n_max_skip
    n_max_skip = int(MAX_SKIP_FRACTION * len(mus))
    n_max_skip = 20
    branch = MatchesMap()
    perform_matching(mus, rec, list(range(len(mus))), list(range(len(rec))), branch)
    print('finished matching')
    potential_matches = sort_potential_matches(potential_matches)
    return potential_matches[0].map

#returns null
def perform_matching(mus, rec, mus_indices, rec_indices, branch):
    global potential_matches
    assert(n_max_skip != 0)
    if branch.count > n_max_skip:
        potential_matches += [branch]
        return
    # if branch has skipped too many notes, prune branch
    potential_matches = filter_potential_matches(potential_matches)
    mus, mus_chord = next_chord(mus)
    # no mus notes left to match, so this branch is complete
    if len(mus_chord) == 0:
        # add remaining notes in rec to skipped notes
        branch.count += len(rec)
        potential_matches += [branch]
        return
    rec_chord = rec[:len(mus_chord)]
    rec = rec[len(rec_chord):]
    mus_chord_indices = mus_indices[:len(mus_chord)]
    rec_chord_indices = rec_indices[:len(rec_chord)]
    mus_indices = mus_indices[len(mus_chord):]
    rec_indices = rec_indices[len(rec_chord):]
    match_update, mus_chord, rec_chord, mus_chord_indices, rec_chord_indices = attempt_chord_matching(
        mus_chord, rec_chord, mus_chord_indices, rec_chord_indices)
    branch = update_branch(branch, match_update)
    # mus and rec chord are different lengths, so should be out of rec notes. Branch is complete
    if len(mus_chord) != len(rec_chord):
        assert (len(rec) == 0)
        branch.count += len(mus_chord) + len(rec_chord)
        potential_matches += [branch]
        return
    # case 1: everything is matched: recurse and ignore other cases. Skipped count not incremented
    assert(len(mus_chord) == len(rec_chord))
    if len(mus_chord) == 0 and len(rec_chord) == 0:
        start = time.time()
        perform_matching(mus, rec, mus_indices, rec_indices, branch)
        dur = time.time() - start
        if dur > 1:
            print('1: {}, remaining: {}'.format(dur, len(mus)))
    else:
        global reached
        if not reached:
            reached = True
        # case 2: delete mus notes, recurse immediately. Add rec notes back to the beginning of rec
        del_mus_branch(mus, rec, mus_chord, rec_chord, rec_chord_indices, mus_indices, rec_indices, branch)
        # case 3: rec notes are extra, delete them until all are matched
        del_rec_branch(mus, rec, mus_chord, rec_chord, mus_chord_indices, mus_indices, rec_indices, branch)
        # case 4: delete everything
        del_both_branch(mus, rec, mus_chord, rec_chord, mus_indices, rec_indices, branch)


def del_mus_branch(mus, rec, mus_chord, rec_chord, rec_chord_indices, mus_indices, rec_indices, branch):
    skipped = len(mus_chord)
    start = time.time()
    perform_matching(
        mus, rec_chord + rec, mus_indices, rec_chord_indices + rec_indices,
        MatchesMap(matches_map=branch).increment(skipped)
    )
    dur = time.time() - start
    if dur > 1:
        print('2: {}, remaining: {}'.format(dur, len(mus)))

def del_rec_branch(mus, rec, mus_chord, rec_chord, mus_chord_indices, mus_indices, rec_indices, branch):
    skipped = 0
    delete_rec_branch = branch
    while len(rec_chord) > 0:
        # delete rec notes
        skipped += len(rec_chord)
        rec_chord = rec[:len(rec_chord)]
        rec = rec[len(rec_chord):]
        rec_chord_indices = rec_indices[:len(rec_chord)]
        rec_indices = rec_indices[len(rec_chord):]
        # attempt to match
        match_update, mus_chord, rec_chord, mus_chord_indices, rec_chord_indices = attempt_chord_matching(
            mus_chord, rec_chord, mus_chord_indices, rec_chord_indices)
        delete_rec_branch = update_branch(branch, match_update)
    start = time.time()
    perform_matching(
        mus, rec, mus_indices, rec_indices, MatchesMap(matches_map=delete_rec_branch).increment(skipped)
    )
    dur = time.time() - start
    if dur > 1:
        print('3: {}, remaining: {}'.format(dur, len(mus)))

def del_both_branch(mus, rec, mus_chord, rec_chord, mus_indices, rec_indices, branch):
    start = time.time()
    skipped = len(mus_chord) + len(rec_chord)
    perform_matching(
        mus, rec, mus_indices, rec_indices, MatchesMap(matches_map=branch).increment(skipped)
    )
    dur = time.time() - start
    if dur > 1:
        print('4: {}, remaining: {}'.format(dur, len(mus)))

def update_branch(branch, update):
    # update is a map of mus index to rec index
    if len(update) == 0:
        return branch
    for key in update:
        assert(key not in branch.map.keys())
    branch.map.update(update)
    return branch

def attempt_chord_matching(mus_chord, rec_chord, mus_indices, rec_indices):
    match_update = {}
    m = 0
    while m < len(mus_chord):
        r = 0
        while r < len(rec_chord):
            if mus_chord[m]['key'] == rec_chord[r]['key']:
                match_update[mus_indices[m]] = rec_indices[r]
                mus_chord = mus_chord[:m] + mus_chord[m+1:]
                mus_indices = mus_indices[:m] + mus_indices[m+1:]
                rec_chord = rec_chord[:r] + rec_chord[r+1:]
                rec_indices = rec_indices[:r] + rec_indices[r+1:]
                m -= 1
                break
            r += 1
        m += 1
    return match_update, mus_chord, rec_chord, mus_indices, rec_indices

def assert_assumptions(mus_chord):
    for i in range(len(mus_chord)):
        for j in range(i+1, len(mus_chord)):
            assert(mus_chord[i]['key'] != mus_chord[j]['key'])

def next_chord(song):
    # returns the next available chord from the beginning of a song
    if len(song) == 0:
        return song, []
    chord_start = song[0]['start']
    chord = []
    it = 0
    while it < len(song) and song[it]['start'] == chord_start:
        chord += [song[it]]
        it += 1
    return song[it:], chord

def chord_generator(song):
    # yields chords from a song
    chord_start = song[0]['start']
    chord = []
    while len(song) != 0:
        it = 0
        while it < len(song) and song[it]['start'] == chord_start:
            chord += [song[it]]
            it += 1
        chord_start = song[it]['start']
        song = song[it:]
        yield chord

def increment_skipped(matches_map_list, skipped):
    for i in range(len(matches_map_list)):
        matches_map_list[i].count += skipped
    return matches_map_list

def sort_potential_matches(potential_matches):
    def sort_fn(elem):
        # elem should be a MatchesMap
        return elem.count
    # should sort from low to high
    potential_matches = sorted(potential_matches, key=sort_fn)
    return potential_matches

def filter_potential_matches(potential_matches):
    potential_matches = sort_potential_matches(potential_matches)
    # cut off matches maps with high skipped counts
    if len(potential_matches) > N_BRANCHES + 10:
        return potential_matches[N_BRANCHES:]
    else:
        return potential_matches

class MatchesMap:
    # just a map with associated data. Data is a count of skipped notes
    def __init__(self, matches_map=None, map=None, count=0):
        if matches_map is None:
            if map is None:
                self.map = {}
            else:
                self.map = map
            self.count = count
        else:
            self.map = copy.deepcopy(matches_map.map)
            self.count = copy.deepcopy(matches_map.count)

    def increment(self, count):
        self.count += count
        return self

