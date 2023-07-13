import numpy as np

def makeLabelDict(path):
    file = open(path, 'r')
    lines = file.readlines()
    
    # nodedict = {}
    nodelist = []
    for line in lines:  
        data = line.split(':')
        index = int(data[0])
        name = str(data[1])[:-1]
        # nodedict[index] = name
        nodelist.append(name)

    return nodelist

def isAttr(idx):
    pass

def getNodeMapping(node_name):
    two_word_names = ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'sports ball', \
                        'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog',
                        'potted plant', 'dining table', 'cell phone', 'teddy bear', 'hair drier', \
                        'light brown']
    two_word_names_correction = ['red-light', 'hydrant', 'stop', 'meter', 'ball', 'bat', 'glove', \
                                'racket', 'wineglass', 'hotdog', 'plant', 'table', 'cellphone', \
                                'teddybears', 'drier', 'light-brown']

    if node_name in two_word_names:
        index = two_word_names.index(node_name)
        return two_word_names_correction[index]
    else:
        return node_name

def readList(path):
    file = open(path, 'r')
    lines = file.readlines()
    # lines = [l.strip()[32:] for l in lines]
    return lines

def isColor(idx):
    nodename2index = makeLabelDict('nodename2index.txt')
    color_list = ['white', 'black', 'blue', 'green', 'red', 'brown', \
                    'yellow', ' gray', 'grey', 'silver', 'grey', 'pink', 'tan']

    is_color = False
    if nodename2index[idx] in color_list: is_color = True

    return is_color

    
