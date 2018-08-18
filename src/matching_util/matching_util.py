import numpy as np
import copy
import time

N_BRANCHES = 10000
N_IN_PROGRESS = 20000
MAX_SKIP_FRACTION = 0.1
# used only for del_rec_branch
N_LOCAL_MAX_SKIP = 20
n_max_skip = 0
completed_branches = []
reached = False
reason_returned = {
    'mus_completed':0,
    'rec_completed':0,
    'del_mus_wrong':0,
    'skipped>local_max_skip':0,
    'del_rec':0,
    'del_both':0,
    'everything_matched':0,
    'skipped>max_skip':0,
}

def get_matching(mus, rec):
    print('beginning matching')
    global completed_branches
    completed_branches = []
    global n_max_skip
    n_max_skip = int(MAX_SKIP_FRACTION * len(mus))
    print(n_max_skip)
    perform_matching([(mus, rec, MatchesMap())])
    print('finished matching')
    completed_branches = sort_completed_branches(completed_branches)
    return completed_branches[0].map

def perform_matching(in_progress_branches):
    global completed_branches
    i = 0
    min_len = float('inf')
    start = time.time()
    while len(in_progress_branches) > 0:
        # prune completed branches
        completed_branches = filter_completed_branches(completed_branches)
        if len(in_progress_branches) > N_IN_PROGRESS:
            in_progress_branches = in_progress_branches[:N_IN_PROGRESS]
        mus, rec, branch = pop_best_branch(in_progress_branches)
        if len(mus) < min_len:
            min_len = len(mus)
        if i % 1000 == 0:
            in_progress_branches = sort_in_progress_branches(in_progress_branches)
            if time.time() - start > 60*60*8 and len(completed_branches) > 0:
                break
            print('In progress: {}, Average mus notes: {}, Furthest progress: {}, Completed: {}, \nReasons: {}\n'.format(
                len(in_progress_branches), np.average(np.asarray([float(len(br[0])) for br in in_progress_branches])),
                min_len, len(completed_branches), reason_returned
            ))
        # filter out long branch
        if branch.count > n_max_skip:
            reason_returned['skipped>max_skip'] += 1
            continue
        new_branches = perform_matching_loop(mus, rec, branch)
        in_progress_branches += new_branches
        i += 1
    print(time.time() - start)

#294 231 764

# returns a list of (mus, rec, branch) tuples to append to in_progress_branches
# returns [] if nothing should be added
def perform_matching_loop(mus, rec, branch, mus_check=(False, None)):
    mus, mus_chord = next_chord(mus)
    # no mus notes left to match, so this branch is complete
    if mus_completed(branch, mus_chord, len(rec)):
        reason_returned['mus_completed'] += 1
        return []
    rec_chord = rec[:len(mus_chord)]
    rec = rec[len(rec_chord):]
    match_update, mus_chord, rec_chord = attempt_chord_matching(mus_chord, rec_chord)
    branch = update_branch(branch, match_update)
    # mus and rec chord are different lengths, so should be out of rec notes. Branch is complete
    if rec_completed(branch, mus_chord, rec_chord, len(rec)):
        reason_returned['rec_completed'] += 1
        return []
    if mus_check[0]:
        prev_rec_indices = mus_check[1]
        cur_rec_indices = [note['index'] for note in rec_chord]
        # check if a previously unmatched rec index is in cur_rec_indices (meaning it's still unmatched)
        # if so, then deleting mus was the wrong assumption. skip this branch
        for r in prev_rec_indices:
            if r in cur_rec_indices:
                reason_returned['del_mus_wrong'] += 1
                return []
    return account_for_error_cases(mus, rec, branch, mus_chord, rec_chord)


def del_mus_branch(mus, rec, mus_chord, rec_chord, branch):
    prev_rec_indices = [note['index'] for note in rec_chord]
    skipped = len(mus_chord)
    branch.count += skipped
    rec = rec_chord + rec
    return perform_matching_loop(mus, rec, branch, mus_check=(True, prev_rec_indices))

# TODO this can be more restrictive
def del_rec_branch(mus, rec, mus_chord, rec_chord, branch):
    skipped = 0
    while len(rec_chord) > 0 and skipped <= N_LOCAL_MAX_SKIP:
        # delete rec notes
        skipped += len(rec_chord)
        rec_chord = rec[:len(rec_chord)]
        rec = rec[len(rec_chord):]
        # attempt to match
        match_update, mus_chord, rec_chord = attempt_chord_matching(mus_chord, rec_chord)
        branch = update_branch(branch, match_update)
        if rec_completed(branch, mus_chord, rec_chord, len(rec)):
            reason_returned['rec_completed'] += 1
            return []
    if skipped > N_LOCAL_MAX_SKIP:
        reason_returned['skipped>local_max_skip'] += 1
        return []
    else:
        reason_returned['del_rec'] += 1
        return [(mus.copy(), rec.copy(), branch.increment(skipped))]

def del_both_branch(mus, rec, mus_chord, rec_chord, branch):
    skipped = len(mus_chord) + len(rec_chord)
    reason_returned['del_both'] += 1
    return [(mus.copy(), rec.copy(), branch.increment(skipped))]

def account_for_error_cases(mus, rec, branch, mus_chord, rec_chord):
    # case 1: everything is matched: recurse and ignore other cases. Skipped count not incremented
    assert (len(mus_chord) == len(rec_chord))
    if len(mus_chord) == 0 and len(rec_chord) == 0:
        reason_returned['everything_matched'] += 1
        return [(mus.copy(), rec.copy(), branch)]
    else:
        new_branches = []
        # case 2: delete mus notes, recurse immediately. Add rec notes back to the beginning of rec
        new_branches += del_mus_branch(mus, rec, mus_chord, rec_chord, MatchesMap(matches_map=branch))
        # case 3: rec notes are extra, delete them until all are matched
        new_branches += del_rec_branch(mus, rec, mus_chord, rec_chord, MatchesMap(matches_map=branch))
        # case 4: delete everything
        new_branches += del_both_branch(mus, rec, mus_chord, rec_chord, MatchesMap(matches_map=branch))
        return new_branches


def update_branch(branch, update):
    # update is a map of mus index to rec index
    if len(update) == 0:
        return branch
    for key in update:
        assert(key not in branch.map.keys())
    branch.map.update(update)
    return branch

def attempt_chord_matching(mus_chord, rec_chord):
    match_update = {}
    m = 0
    while m < len(mus_chord):
        r = 0
        while r < len(rec_chord):
            if mus_chord[m]['key'] == rec_chord[r]['key']:
                match_update[mus_chord[m]['index']] = rec_chord[r]['index']
                mus_chord = mus_chord[:m] + mus_chord[m+1:]
                rec_chord = rec_chord[:r] + rec_chord[r+1:]
                m -= 1
                break
            r += 1
        m += 1
    return match_update, mus_chord, rec_chord

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

#returns True if no more mus notes, so branch completed. Does associated logic
# note that these two methods take care of adding to completed branches. if returns true, in_progress_branches is not updated
def mus_completed(branch, mus_chord, rec_len):
    global completed_branches
    # no mus notes left to match, so this branch is complete
    if len(mus_chord) == 0:
        # add remaining notes in rec to skipped notes
        branch.count += rec_len
        completed_branches += [branch]
        return True
    else:
        return False

# returns True if no more rec notes
def rec_completed(branch, mus_chord, rec_chord, rec_len):
    global completed_branches
    # mus and rec chord are different lengths, so should be out of rec notes. Branch is complete
    if len(mus_chord) != len(rec_chord):
        assert (rec_len == 0)
        branch.count += len(mus_chord) + len(rec_chord)
        completed_branches += [branch]
        return True
    else:
        return False

def increment_skipped(matches_map_list, skipped):
    for i in range(len(matches_map_list)):
        matches_map_list[i].count += skipped
    return matches_map_list

def sort_in_progress_branches(branches):
    def sort_fn(elem):
        # elem should be a MatchesMap
        return elem[2].count / (len(elem[2].map) + 1)
    # should sort from low to high
    branches = sorted(branches, key=sort_fn, reverse=True)
    return branches

def sort_completed_branches(potential_matches):
    def sort_fn(elem):
        # elem should be a MatchesMap
        return elem.count
    # should sort from low to high
    potential_matches = sorted(potential_matches, key=sort_fn)
    return potential_matches

def filter_completed_branches(branches_list):
    # cut off matches maps with high skipped counts
    if len(branches_list) > N_BRANCHES + 10:
        branches_list = sort_completed_branches(branches_list)
        branches_list = branches_list[:N_BRANCHES]
        assert(len(branches_list) == N_BRANCHES)
        return branches_list
    else:
        return branches_list

def pop_best_branch(in_progress_branches, last=True):
    if last:
        return in_progress_branches.pop(-1)
    else:
        index = -1
        best_count = 1.0
        for i in range(len(in_progress_branches)):
            if len(in_progress_branches[i][2].map) == 0:
                index = -1
                break
            if in_progress_branches[i][2].count / len(in_progress_branches[i][2].map) < best_count:
                best_count = in_progress_branches[i][2].count / len(in_progress_branches[i][2].map)
                index = i
        mus, rec, branch = in_progress_branches.pop(index)
        return mus, rec, branch

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

    def __len__(self):
        return len(self.map)

