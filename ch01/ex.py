import sys
sys.path.append('..')

from dataset import spiral
import matplotlib.pyplot  as plt

x, t = spiral.load_data()
print('x', x[0:10, :])
print('t', t[0:10, :])