import sys
sys.path.append('..')

import numpy as np
from common.util import *
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

print('calculating SVD ...')
# try:
from sklearn.utilsextmath import randomized_svd
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, rando_state=None)
# except ImportError:
#     U, S, V = np.ligalg.svd(W)

word_vecs = U[:, :wordvec_size]
querys = ['you', 'year', 'car', 'love', 'honda']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)