import sys
sys.path.append('..')

from common.np import *
from common.config import GPU
from common.functions import softmax, cross_entropy_error

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        # print('sigmoid forward: x {}, out: {}'.format(x.shape, out.shape))
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        # print('sigmoid backward: dout: {}, dx: {}'.format(dout.shape, dx.shape))
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
    # hoho_todo

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        
        loss = cross_entropy_error(self.y, self.t)

        # print('SoftmaxWithLoss forward: x {}, t {}, loss {}'.format(x.shape, t.shape, loss))

        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        # print('SoftmaxWithLoss backward: dx {}'.format(dx.shape))

        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x

        # print('Affine forward: W {}, b {}, x {}, out {}'.format(W.shape, b.shape, x.shape, out.shape))
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        # print('Affine backward: W {}, dW {}, b {}, db {}, dx {}'.format(W.shape, dW.shape, b.shape, db.shape, dx.shape))

        return dx

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(w)]
        self.x = None
    
    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):   # dout为上游传过来的梯度
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


if __name__ == '__main__':
    print('done')