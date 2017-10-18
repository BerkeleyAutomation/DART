import numpy as np
import gym
from expert import tf_util

class GaussianSupervisor():

    def __init__(self, policy, cov):
        self.policy = policy
        self.cov = cov

    def sample_action(self, s):
        intended_action = self.policy.intended_action(s)
        sampled_action = np.random.multivariate_normal(intended_action, self.cov)
        return sampled_action


    def intended_action(self, s):
        return self.policy.intended_action(s)

class Supervisor():

    def __init__(self, policy_fn, sess):
        self.policy_fn = policy_fn
        self.sess = sess
        with self.sess.as_default():
            tf_util.initialize()

    def sample_action(self, s):
        with self.sess.as_default():
            intended_action = self.policy_fn(s[None,:])[0]
            return intended_action

    def intended_action(self, s):
        return self.sample_action(s)


class EpsSupervisor():
    """
        Discrete action space
    """
    def __init__(self, policy, action_space, eps = 0.0):
        self.policy = policy
        self.eps = eps
        self.action_space = action_space

    def sample_action(self, s):
        intended_action = self.policy.intended_action(s)
        if np.random.uniform(0, 1) > self.eps:
            return intended_action
        else:
            return np.random.choice(self.action_space)


    def intended_action(self, s):
        return self.policy.intended_action(s)
