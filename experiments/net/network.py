import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython
import pickle
import os
from numba import jit
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




class Network:

    def __init__(self, arch, learning_rate=.01, epochs=40, mean=None, std=None):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.arch = np.array(arch)
        self.learning_rate = learning_rate
        self.constructed = False
        self.bsize = 200
        self.iters = epochs * 10
        self.mean = mean
        self.std = std

    @jit
    def whiten(self, X):
        X = X - self.mean
        X = X / self.std
        locs = np.isnan(X)
        X[locs] = 0.0
        locs = np.isinf(X)
        X[locs] = 0.0
        return X
     

    @jit
    def shuffle(self, X, y):
        p = np.random.permutation(X.shape[0])
        return X[p], y[p]

    @jit
    def params(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)


    def fit(self, X, y):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        X, y = np.array(X), np.array(y)
        self.params(X)

        X = self.whiten(X)
        n = X.shape[1]
        d = y.shape[1]
        if not self.constructed:
            self.construct(n, d)
            self.constructed = True

        # X, y = self.shuffle(X, y)
        # cutoff = int(X.shape[0] * .05)
        # X_test, y_test = X[:cutoff, :], y[:cutoff, :]
        # X, y = X[cutoff:, :], y[cutoff:, :]

        X, X_test, y, y_test = train_test_split(X, y, test_size=.05)

        losses = []
        val_losses = []

        with self.graph.as_default():
            for i in range(self.iters):

                X, y = self.shuffle(X, y)
                j = np.random.randint(X.shape[0] - self.bsize)

                fd = {self.states: X[j:j+self.bsize, :], self.actions: y[j:j+self.bsize, :]}
                _, loss = self.sess.run([self.opt, self.loss], feed_dict = fd)


                if i == (self.iters - 1) or self.iters > 3 and i % (self.iters / 3) == 0:
                    print "\t iter: " + str(i)
                    print bcolors.WARNING + "\t Loss: " + str(loss) + bcolors.ENDC
                    losses.append(loss)
                if i == (self.iters - 1) or self.iters > 3 and i % (self.iters / 3) == 0:
                    fd = {self.states: X_test, self.actions: y_test}
                    val_loss = self.sess.run(self.loss, feed_dict=fd)
                    val_losses.append(val_loss)
                    print bcolors.OKGREEN + "\t Valid loss: " + str(val_loss) + bcolors.ENDC

        return {"loss": losses, "val_loss": val_losses}


    def predict(self, X):
        X = np.array(X)
        X = self.whiten(X)

        with self.graph.as_default():
            with self.sess.as_default():
                fd = {self.states: X}
                preds = self.sess.run(self.outputs, feed_dict = fd)
        return preds


    def score(self, X, y):
        X = self.whiten(X)
        X, y = np.array(X), np.array(y)
        with self.graph.as_default():
            fd = {self.states: X, self.actions: y}
            return self.sess.run(self.r_squared, feed_dict = fd)


    def construct(self, n, d):
        with self.graph.as_default():
            with self.sess.as_default():

                arch = np.insert(self.arch, 0, n)

                self.states = tf.placeholder("float", [None, n])
                self.actions = tf.placeholder("float", [None, d])

                buffer_layer = self.states
                # if self.mean is not None and self.std is not None:
                #     buffer_layer = (buffer_layer - self.mean) / (self.std + 1e-6)

                layers = [buffer_layer]
                last_layer = buffer_layer

                weights = []
                biases = []

                for i in range(len(arch))[1:]:
                    size = arch[i]
                    prev_size = arch[i - 1]
                    w1 = tf.Variable(tf.random_normal([prev_size, size], stddev=.15), name='w' + str(i))
                    b1 = tf.Variable(tf.random_normal([size], stddev=.15), name='b' + str(i))
                    weights.append(w1)
                    biases.append(b1)
                    layer = tf.nn.tanh(tf.matmul(layers[i-1], w1) + b1)
                    layers.append(layer)

                w1 = tf.Variable(tf.random_normal([arch[-1], d], stddev=.15), name='w' + str(len(arch) - 1))
                b1 = tf.Variable(tf.random_normal([d], stddev=.15), name='b' + str(len(arch) - 1))
                weights.append(w1)
                biases.append(b1)
                self.weights = weights
                self.biases = biases

                self.outputs = tf.matmul(layers[-1], w1) + b1

                self.loss = tf.reduce_mean(tf.square(self.outputs - self.actions))
                self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

                self.sum_loss = tf.reduce_sum(tf.square(self.outputs - self.actions))
                self.r_squared = 1 - self.sum_loss / (tf.reduce_sum(tf.square(self.actions - tf.reduce_mean(self.actions, 0))))

                self.sess.run(tf.global_variables_initializer())




    @staticmethod
    def norms(filename):
        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())
        nonlin_type = data['nonlin_type']
        # print nonlin_type
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
        policy_params = data[policy_type]

        assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))

        return obsnorm_mean, obsnorm_stdev


if __name__ == '__main__':
    net = Network([10])

    start, end = 0, 2* np.pi
    n = 75
    n_test = 50

    X = np.random.uniform(start, end, (n, 1))
    Y = (np.sin(X) + np.random.normal(0, .31, (n, 1))).reshape(n, 1)

    X_test = np.random.uniform(start+.4, end-.4, (n_test, 1))
    Y_test = (np.sin(X_test) + np.random.normal(0, .321, (n_test, 1))).reshape(n_test)


    net.fit(X, Y)

    x = np.arange(start, end, .1)
    n = len(x)
    x = x.reshape((n, 1))
    preds = net.predict(x)

    print "Score: " + str(net.score(X, Y))


    print X.shape
    print Y.shape
    plt.scatter(X_test[:, 0], Y_test)
    plt.ylim(-5 ,5)
    plt.plot(x, np.sin(x).reshape(n),  color='r', linestyle='--')
    plt.plot(x, preds.reshape(n), color='g')
    plt.show()


