import sys
sys.path.append('..')

import numpy as np
from common.layers import Affine, Sigmoid


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

if __name__ == '__main__':
    # x = np.random.randn(10, 2)
    # model = TwoLayerNet(2, 4, 3)
    # s = model.predict(x)

    # x = np.array([[1, 2, 3, 4], [5, 6, 7, 9]])
    # print(x.max(), end='\n')
    # print(x.max(axis=1), end='\n')
    # print(x.max(axis=1, keepdims=True), end='\n')
    # print(x.size)

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    # a = b
    # a[...] = b
    print(id(a), id(b))
    b *= 2
    print(b)
