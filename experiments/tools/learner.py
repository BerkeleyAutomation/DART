import numpy as np
from sklearn.model_selection import train_test_split
"""
    sup should have two methods: intended_action() and sample_action()
    which return the intended action and the potentially noisy action respectively.
"""

class Learner():

    def __init__(self, est, sup=None):
        self.X = []
        self.y = []
        self.est = est
        self.one_class_error = None

    def add_data(self, states, actions):
        assert type(states) == list
        assert type(actions) == list
        self.X += states
        self.y += actions

    def clear_data(self):
        self.X = []
        self.y = []

    def train(self, verbose=False):
        # try:
        if True:
            # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.15)
            X_train, y_train = self.X, self.y
            history = self.est.fit(X_train, y_train)
            self.one_class_error = None
            if verbose == True:
                print "Train score: " + str(self.est.score(X_train, y_train))
            return history

            # return self.est.score(X_test, y_test)
        # except ValueError:
            # self.one_class_error = self.y[0]

    def acc(self):
        if self.one_class_error is not None:
            predictions = np.ones(len(self.y)) * self.one_class_error
            return np.mean((predictions == np.array(self.y)).astype(int))
            
        return self.est.score(self.X, self.y)

    def intended_action(self, s):
        if self.one_class_error is not None:
            return self.one_class_error
        return self.est.predict([s])[0]

    def sample_action(self, s):
        return self.intended_action(s)



