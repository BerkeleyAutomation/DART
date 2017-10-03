import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import losses
from keras import backend
import matplotlib.pyplot as plt
import os
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

    def __init__(self, arch, learning_rate=.01, epochs=600):
        backend.clear_session()
        self.constructed = False
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.arch = arch
        self.bsize = 128
        self.model = None
        self.mean = None
        self.std = None

    def whiten(self, X):
        X = X - self.mean
        X = X / self.std
        locs = np.isnan(X)
        X[locs] = 0.0
        locs = np.isinf(X)
        X[locs] = 0.0
        return X

    def params(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)


    def fit(self, X, y, X_test=None, y_test=None):
        X, y = np.array(X), np.array(y)
        self.params(X)
        X = self.whiten(X)
        n = X.shape[1]
        d = y.shape[1]
        # if not self.constructed:
        self.construct(n, d)
        self.constructed = True

        if X_test is None or y_test is None:
            X, X_test, y, y_test = train_test_split(X, y, test_size=.05)
            history = self.model.fit(X, y, batch_size=self.bsize, epochs = self.epochs, 
                    verbose=0, validation_data=(X_test, y_test))
            print "Val_loss: " + str(history.history['val_loss'][-1])
            print "loss: " + str(history.history['loss'][-1])
        else:
            X_test, y_test = np.array(X_test), np.array(y_test)
            history = self.model.fit(X, y, batch_size=self.bsize, epochs = self.epochs, 
                    verbose=0, validation_data=(X_test, y_test))

        return history.history

    def predict(self, X):
        X = np.array(X)
        X = self.whiten(X)
        preds = self.model.predict(X, batch_size = 128, verbose=0)
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        loss = np.sum(np.square(y - preds))
        res = np.sum(np.square(y - np.mean(y, axis=0)))
        return 1.0 - loss/res

    def construct(self, n, d):
        backend.clear_session()
        model = Sequential()
        last_shape = (n,)
        for i in range(len(self.arch)):
            model.add(Dense(self.arch[i], activation='tanh', input_shape=last_shape))
            last_shape = (self.arch[i],)

        model.add(Dense(d, activation='linear', input_shape=last_shape))
        model.compile(loss=losses.mean_squared_error,
              optimizer=optimizers.Adam(lr=self.learning_rate))
        self.model = model


if __name__ == '__main__':
    net = Network([200, 100, 50])

    start, end = 0, 2* np.pi
    n = 75
    n_test = 50

    X = np.random.uniform(start, end, (n, 1))
    Y = (np.sin(X) + np.random.normal(0, .031, (n, 1))).reshape(n, 1)

    X_test = np.random.uniform(start+.4, end-.4, (n_test, 1))
    Y_test = (np.sin(X_test) + np.random.normal(0, .0321, (n_test, 1))).reshape(n_test)


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


