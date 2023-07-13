from keras.preprocessing.text import Tokenizer
from utils.data_utils import makeLabelDict
import numpy as np

def data_preprocessing(data):
    print (len(data))
    oov_token = "<OOV>"
    vocab_size = len(data)
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(data)
    word_index = tokenizer.word_index
    return word_index

def print_two_words_nodes(data, word_index):
    for word in data:
        if len(word.split(' ')) > 1:
            print (word)

def use_glove_embeddings():
    embeddings_index = {}
    f = open('./pretrained_embeddings/glove.6B.100d.txt')
    print ('Loading GloVe embeddings ...')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

if (__name__ == '__main__'):

    index2node = makeLabelDict('nodename2index.txt')
    word_index = data_preprocessing(index2node)
    print_two_words_nodes(index2node, word_index)
    embeddings_index = use_glove_embeddings()
    # print (sorted(embeddings_index.keys()))
    import pdb
    pdb.set_trace()
    print (embeddings_index.get('dining-table'))
    # embedding_matrix = vocab2glove(word_index, embeddings_index)
    # print (embedding_matrix.shape)