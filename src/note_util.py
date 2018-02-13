def denormalizeTimes(predictions, lastTime):
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            predictions[i][j] *= (lastTime / 100.0)
    return predictions

def denormalizeTimeFromOffsetSubtract(offset_norm, original, last_mus, last_rec):
    return offset_norm * last_rec / 100.0 + original * last_rec / last_mus

# sets starting and ending times to be percentage of total song length
def normalizeTimes(musList, recList):
    for i in range(len(musList)):
        lastTime = float(musList[i][-1]['end'])
        for note in musList[i]:
            note['start_normal'] = (note['start'] * 100.0) / lastTime
            note['end_normal'] = (note['end'] * 100.0) / lastTime
    for i in range(len(recList)):
        lastTime = float(recList[i][-1]['end'])
        for note in recList[i]:
            note['start_normal'] = (note['start'] * 100.0) / lastTime
            note['end_normal'] = (note['end'] * 100.0) / lastTime
            #note['offset_normal'] = (note['offset'] * 100.0) / lastTime
            #note['len_offset_normal'] = (note['len_offset'] * 100.0) / lastTime
    return musList, recList

def normalizeIndices(musList, recList):
    for i in range(len(musList)):
        lastIndex = float(musList[i][-1]['index'])
        for note in musList[i]:
            note['index_normal'] = (note['index'] * 100) / lastIndex
    for i in range(len(recList)):
        lastIndex = float(recList[i][-1]['index'])
        for note in recList[i]:
            note['index_normal'] = (note['index'] * 100) / lastIndex
    return musList, recList

def denormalizeIndices(indices, lastIndex):
    for i in range(len(indices)):
        indices[i] *= (lastIndex / 100.0)
        indices[i] = int(indices[i])
    return indices

def denormalizeIndex(index, lastIndex):
    return int(index * lastIndex / 100.0)

def addOffsets(musList, recList):
    for i in range(len(musList)):
        mus = musList[i]
        rec = recList[i]
        for j in range(len(mus)):
            m = mus[j]
            r = rec[j]
            r['offset'] = r['start'] - m['start']
            r['len_offset'] = (r['end'] - r['start']) - (m['end'] - m['start'])
            r['offset_normal'] = r['start_normal'] - m['start_normal']
            r['len_offset_normal'] = (r['end_normal'] - r['start_normal']) - (m['end_normal'] - m['start_normal'])
            recList[i][j] = r
    return recList


def readPIDI(path):
    with open(path) as f:
        lines = f.readlines()
    if lines[0] == "NAME":
        name = lines[1]
        lines = lines[2:]
    else:
        name = None
        lines = lines[1:]
    song = []
    for line in lines:
        song.append(decodeNote(line))
    return name, song


def writePIDI(song, path, name=None):
    with open(path, 'w') as of:
        if name is not None:
            of.write("NAME\n")
            of.write(name + "\n")
        else:
            of.write("NO NAME\n")
        for note in song:
            of.write(printNote(note) + "\n")


def decodeNote(str):
    end = str.find("}")
    if end == -1:
        end = len(str)
    str = str[str.find("{") + 1: end]
    note = {}
    arr = str.split(',')
    note['key'] = int(arr[0])
    note['index'] = int(arr[1])
    note['onv'] = int(arr[2])
    note['offv'] = int(arr[3])
    note['start'] = int(arr[4])
    note['end'] = int(arr[5])
    if note['end'] < 0 or note['start'] < 0:
        print(arr)
    note['track'] = int(arr[6])
    return note


def printNote(note):
    str = "{{{},{},{},{},{},{},{}}}".format(note['key'], note['index'], note['onv'], note['offv'],
                                            note['start'], note['end'], note['track'])
    return str