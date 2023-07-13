import numpy as np
from scipy.spatial.distance import cdist

emb_dim = 50

def use_glove_embeddings():
    embeddings_index = {}
    f = open('./pretrained_embeddings/glove.6B.{}d.txt'.format(emb_dim))
    # print ('Loading GloVe embeddings ...')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def vocab2glove(word_index_list, embeddings_index):
    max_length = emb_dim
    embedding_matrix = np.zeros((len(word_index_list), max_length))
    
    #### if word_index is a dict
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i - 1] = embedding_vector
    # return embedding_matrix
    
    for i, word in enumerate(word_index_list):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_embedding_matrix(word_list, embeddings_index):
    embedding_matrix = vocab2glove(word_list, embeddings_index)
    return embedding_matrix

def get_embedding_matrix_database(embeddings_index):    
    max_length = emb_dim
    embedding_matrix = np.zeros((len(embeddings_index.items()), max_length))
    database_word_list = []
    
    i = 0
    for word, embedding_vector in embeddings_index.items():
        embedding_matrix[i] = embedding_vector
        database_word_list.append(word)
        i += 1

    return embedding_matrix, database_word_list
    
def compute_closest(node_emb, embedding_matrix_database, dis_thresh=2.5):
    '''
    node_emb: embedding of node to compute distance from
    embedding_matrix_database: matrix containing all embeddings for all words in glove database
    dis_thresh: threshold distance to consider a neighbor
    '''
    distances_from_curr = cdist(embedding_matrix_database, node_emb.reshape(1, -1))[:, 0]
    neighbor_idx = np.argwhere(distances_from_curr < dis_thresh)
    embed_closest = []
    for idx in neighbor_idx: embed_closest.append(embedding_matrix_database[idx])
    return neighbor_idx, embed_closest