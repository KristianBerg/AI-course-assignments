import re
import numpy as np


def input_file(filename):
    with open(filename) as file:
        arr = []
        pattern = re.compile(r'(\d+)\s+(\d+)')
        for line in file:
            matches = pattern.match(line)
            if matches:
                first, second = matches.groups()
                arr.append([1, float(first), float(second)])
        return arr


def add_class(class_label, iterable):
    return list(map(lambda x: (class_label, x), iterable))

def to_batch(classed_list):
    labels, feats = map(lambda x: np.array(x), zip(*classed_list))
    return (labels.T, feats.T)

def format_LIBSVM(lst):
    str = ''
    for example in lst:
        str += repr(example[0])
        for index in range(1,len(example[1])):
            feature = example[1][index]
            str += ' ' + repr(index) + ':' + repr(feature)
        str += '\n'
    return str[0:-1]


def write_LIBVSM(lst, filename):
    with open(filename, 'w+') as file:
        file.write(formatLIBSVM(lst))
        
    
def read_LIBVSM(iterable, num_feat):
    label = re.compile(r'(\d+)')
    feature = re.compile(r'\s*(\d+:\d+)')
    arr = []
    for line in iterable:
        mat = label.match(line)
        features = [1] + [0] * num_feat
        for item in feature.finditer(line, mat.end()):
            index, value = item.group().split(':')
            features[int(index)] = float(value)
        arr.append((int(mat.group()), features))
    return arr


def input_LIBVSM(filename, num_feat):
    with open(filename) as file:
        return read_LIBVSM(file, num_feat)

    
def input_dense_LIBVSM(filename):
    with open(filename) as file:
        num_feat = len(re.findall(':', file.readline()))
        file.seek(0)
        return read_LIBVSM(file, num_feat)
