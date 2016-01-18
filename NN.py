from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import time


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NNetwork(object):
    def __init__(self, hidden=1):
        self.input_layer_size = 2
        self.hidden_layer_size = hidden
        self.output_layer_size = 1
        self.param_size = self.input_layer_size * self.hidden_layer_size + \
                          self.hidden_layer_size * self.output_layer_size
        self.set_param(np.random.randn(self.param_size, 1))

    def set_param(self, vector):
        sep = self.input_layer_size * self.hidden_layer_size
        self.w1 = vector.ravel()[:sep].reshape(self.input_layer_size, self.hidden_layer_size)
        self.w2 = vector.ravel()[sep:].reshape(self.hidden_layer_size, self.output_layer_size)

    def get_param(self):
        return np.concatenate([self.w1.ravel(), self.w2.ravel()])

    def forward(self, xx):
        self.a1 = xx
        self.z2 = np.dot(self.a1, self.w1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = sigmoid(self.z3)
        return self.a3

    def backward(self, yh, yy):
        self.delta3 = (self.a3 - yy) * sigmoid_grad(self.z3)
        grad2 = np.dot(self.a2.T, self.delta3)
        self.delta2 = self.delta3 * self.w2.T * sigmoid_grad(self.z2)
        grad1 = np.dot(self.a1.T, self.delta2)
        grad = np.concatenate([grad1.ravel(), grad2.ravel()])
        return grad

    def cost_func(self, ww, xx, yy, lr):
        self.set_param(ww)
        y_hat = self.forward(xx)
        grad = self.backward(y_hat, yy).T * lr
        j_cost = np.dot((yy - y_hat).T, (yy - y_hat)) / 2
        return j_cost, grad


class Trainer(object):
    def __init__(self, network, xx, yy):
        self.nn = network
        self.init_point = self.nn.get_param()
        self.x = xx
        self.y = yy

    # time estimator
    class Timer(object):
        def __init__(self, name=None):
            self.name = name

        def __enter__(self):
            self.tstart = time.time()

        def __exit__(self, type, value, traceback):
            if self.name:
                print '[%s]' % self.name,
            print 'Elapsed: %s\n' % (time.time() - self.tstart)

    def callback(self, pp):
        j = self.nn.cost_func(pp, self.x, self.y, self.lr)[0]
        self.j_hist.extend(j[0])
        if self.show_flag:
            print j

    def execute(self, learning_rate=3, iter_max=100, show_flag=False, new_network=None):
        if new_network:
            t.__init__(new, self.x, self.y)
        self.show_flag = show_flag
        self.lr = learning_rate
        self.iter_max = iter_max
        self.j_hist = []
        options = {'maxiter': self.iter_max, 'disp': True}
        print "learning rate:%.2f, max iteration: %d" % (self.lr, self.iter_max)
        with self.Timer("test minimizing time"):
            self.res = minimize(self.nn.cost_func, self.init_point, args=(self.x, self.y, self.lr), jac="True",
                                callback=self.callback,
                                method='BFGS', options=options)


if __name__ == "__main__":
    # data
    x = np.array([[3, 5], [5, 1], [10, 2]])
    y = np.array([[.75, ], [.82, ], [.93, ]])

    # init network & parameters
    n = NNetwork(2)

    # training
    t = Trainer(n, x, y)
    t.execute()

    # show result
    n.set_param(t.res.x)
    print n.forward(x).T
    print y.T

    # plot 4 different learning rate
    lr_list = [[0.1, 0.3], [1, 3]]
    f, ax = plt.subplots(2, 2)
    for rr in xrange(len(lr_list)):
        for r in xrange(len(lr_list[rr])):
            t.execute(lr_list[rr][r])
            ax[rr][r].plot(t.j_hist)
            ax[rr][r].set_title('learning rate = %.1f' % lr_list[rr][r])
            ax[rr][r].set_xlim([0, 100])
            # ax[rr][r].set_ylim([0, 0.25])
    plt.show()

    # plot 4 different learning rate
    hidden_list = [[1, 3], [5, 10]]
    f, ax = plt.subplots(2, 2)
    for hh in xrange(len(hidden_list)):
        for h in xrange(len(hidden_list[hh])):
            new = NNetwork(hidden_list[hh][h])
            t.execute(new_network=new)
            ax[hh][h].plot(t.j_hist)
            ax[hh][h].set_title('hidden layer size:%d' % hidden_list[hh][h])
            ax[hh][h].set_xlim([0, 100])
    plt.show()

    # grad check
    # if False:
    #     num_grad = np.zeros(n.param_size)
    #     e = 1e-4
    #     count = 0
    #     for p in param0:
    #         p += e
    #         t1 = n.cost_func(param0, x, y)[0]
    #         p -= 2 * e
    #         t2 = n.cost_func(param0, x, y)[0]
    #         p += e
    #         num_grad[count] = (t1 - t2) / (2 * e)
    #         count += 1
    #     print num_grad
    #
