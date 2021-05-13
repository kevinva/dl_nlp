import sys
sys.path.append('..')

from dataset import spiral
import matplotlib.pyplot  as plt
import numpy as np

# x, t = spiral.load_data()
# print('x', x[0:10, :])
# print('t', t[0:10, :])
y = np.random.randn(10, 3)
t = y.argmax(axis=1)
print(y)
# print(y.argmax(axis=1))
print(t)
print(y[np.arange(10), t])