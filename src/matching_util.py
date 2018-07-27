import numpy as np

N_BRANCHES = 100

def get_matching(mus, rec):
    print('beginning matching')
    potential_matches = perform_matching(mus, rec, list(range(len(mus))), list(range(len(rec))), [])
    print('finished matching')
    return sort_potential_matches(potential_matches)[0].map

def perform_matching(mus, rec, mus_indices, rec_indices, potential_matches):
    print('matching next chord')
    mus, mus_chord = next_chord(mus)
    if len(mus_chord) == 0:
        return potential_matches
    rec_chord = rec[:len(mus_chord)]
    if len(mus_chord) != len(rec_chord):
        print('hi')
    assert (len(mus_chord) == len(rec_chord))
    rec = rec[len(rec_chord):]
    mus_chord_indices = mus_indices[:len(mus_chord)]
    rec_chord_indices = rec_indices[:len(rec_chord)]
    mus_indices = mus_indices[len(mus_chord):]
    rec_indices = rec_indices[len(rec_chord):]
    match_update, mus_chord, rec_chord, mus_chord_indices, rec_chord_indices = attempt_chord_matching(
        mus_chord, rec_chord, mus_chord_indices, rec_chord_indices)
    potential_matches = update_potential_matches(potential_matches, match_update)
    match_update_list = []
    # case 1: everything is matched: recurse and ignore other cases. Skipped count not incremented
    assert(len(mus_chord) == len(rec_chord))
    if len(mus_chord) == 0:
        potential_matches = perform_matching(mus, rec, mus_indices, rec_indices, potential_matches)
        return potential_matches
    else:
        # case 2: delete mus notes, recurse immediately. Add rec notes back to the beginning of rec
        skipped = len(mus_chord)
        potential_matches_branch1 = perform_matching(
            mus, rec_chord + rec, mus_indices, rec_chord_indices + rec_indices,
            increment_skipped(potential_matches, skipped)
        )
        # case 3: rec notes are extra, delete them until all are matched
        potential_matches_branch2 = potential_matches
        skipped = 0
        while len(rec_chord) > 0:
            # delete rec notes
            skipped += len(rec_chord)
            rec_chord = rec[:len(rec_chord)]
            rec = rec[len(rec_chord):]
            rec_chord_indices = rec_chord_indices[:len(rec_chord)]
            rec_indices = rec_indices[len(rec_chord):]
            # attempt to match
            match_update, mus_chord, rec_chord, mus_chord_indices, rec_chord_indices = attempt_chord_matching(
                mus_chord, rec_chord, mus_chord_indices, rec_chord_indices)
            potential_matches_branch2 = update_potential_matches(potential_matches_branch2, match_update)
        potential_matches_branch2 = perform_matching(
            mus, rec, mus_indices, rec_indices, increment_skipped(potential_matches_branch2, skipped)
        )
        potential_matches = potential_matches_branch1 + potential_matches_branch2
        potential_matches = filter_potential_matches(potential_matches)
        return potential_matches



def update_potential_matches(potential_matches, update):
    # update is a map of mus index to rec index
    if len(update) == 0:
        return potential_matches
    if len(potential_matches) == 0:
        potential_matches = [MatchesMap(map=update, count=0)]
        return potential_matches
    for i in range(len(potential_matches)):
        for key in update:
            if key in potential_matches[i].map.keys():
                print('hi')
        potential_matches[i].map.update(update)
    return potential_matches

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
    def __init__(self, map=None, count=0):
        if map is None:
            self.map = {}
        else:
            self.map = map
        self.count = count

