import numpy as np

# print(np.random.choice(10))

words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
for i in range(20):
    print(np.random.choice(words, p=p))