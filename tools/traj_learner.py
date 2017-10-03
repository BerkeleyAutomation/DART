from learner import Learner
import numpy as np

class Learner(Learner):

    def __init__(self, est, sup):
        self.state_trajs = []
        self.action_trajs = []
        self.est = est
        self.sup = sup
        self.one_class_error = None
        self.cutoff = .1

    def add_data(self, states, actions):
        self.state_trajs.append(states)
        self.action_trajs.append(actions)


    def acc(self):
        X = self.make2d(self.state_trajs)
        y = self.make2d(self.action_trajs)
        return self.est.score(X, y)


    def train(self, verbose=False):
        trajs = zip(self.state_trajs, self.action_trajs)
        np.random.shuffle(trajs)
        self.state_trajs, self.action_trajs = zip(*trajs)
        self.state_trajs, self.action_trajs = list(self.state_trajs), list(self.action_trajs)

        X, y, X_test, y_test = None, None, None, None

        cutoff = int(round(self.cutoff * len(self.state_trajs)))
        try: 
            X = self.make2d(self.state_trajs[cutoff:])
        except:
            IPython.embed()
        y = self.make2d(self.action_trajs[cutoff:])

        if not cutoff == 0:
            X_test = self.make2d(self.state_trajs[:cutoff])
            y_test = self.make2d(self.action_trajs[:cutoff])

        history = self.est.fit(X, y, X_test, y_test)
        self.one_class_error = None
        if verbose == True:
            print "Train score: " + str(self.est.score(X, y))
            if X_test is not None:
                print "Test score: " + str(self.est.score(X_test, y_test))
        return history

    def make2d(self, data):
        result = np.array(data[0])
        for traj in data[1:]:
            result = np.vstack((result, traj))
        return result
