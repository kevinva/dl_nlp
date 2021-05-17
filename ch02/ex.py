import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from common.util import *

text = 'You say goodbye and I say hello.'

# text = text.lower()
# text = text.replace('.', ' .')
# print(text)

# words = text.split(' ')
# word_to_id = {}
# id_to_word = {}

# for word in words:
#     if word not in word_to_id:
#         new_id = len(word_to_id)
#         word_to_id[new_id] = word
#         id_to_word[word] = new_id

# print(word_to_id)
# print(id_to_word)

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
# most_similar('and', word_to_id, id_to_word, C, top=5)
W = ppmi(C)
np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-' * 50)
print('PPMI')
print(W)

U, S, V = np.linalg.svd(W)
print('-' * 50)
print('U')
print(U)
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()