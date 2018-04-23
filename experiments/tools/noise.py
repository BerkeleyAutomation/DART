import numpy as np
import statistics
import IPython

def sample_covariance_lnr(env, lnr, sup, samples, T):

    cov = np.zeros(env.action_space.shape[0])
    for s in range(samples):
        states, tmp_actions, _, _ = statistics.collect_traj(env, lnr, T)
        sup_actions = np.array([sup.intended_action(s) for s in states])
        lnr_actions = np.array(tmp_actions)

        length = len(tmp_actions)
        diff = sup_actions - lnr_actions

        cov = cov + np.dot(diff.T, diff) / float(length)
    
    return cov / float(samples)


def sample_covariance_sup(env, lnr, sup, samples, T):
    cov = np.zeros(env.action_space.shape[0])
    for s in range(samples):
        states, tmp_actions, _, _ = statistics.collect_traj(env, sup, T)
        sup_actions = np.array(tmp_actions)
        lnr_actions = np.array([lnr.intended_action(s) for s in states])
        length = len(tmp_actions)

        diff = sup_actions - lnr_actions
        cov = cov + np.dot(diff.T, diff) / float(length)

    return cov / float(samples)


def sample_covariance_trajs(env, lnr, trajs, samples, T):
    d = env.action_space.shape[0]
    cov = np.zeros((d, d))
    trajs = np.array(trajs[len(trajs) - samples:])
    # trajs = np.array(trajs[:])
    # trajs = np.array(trajs[len(trajs) - samples * 2:])
    # indices = np.random.choice(len(trajs), min(len(trajs), samples), replace=False)
    # trajs = trajs[indices]
    for states, i_actions in trajs:
        sup_actions = np.array([a for a in i_actions])
        lnr_actions = np.array([lnr.intended_action(s) for s in states])
        length = len(i_actions)

        diff = sup_actions - lnr_actions
        cov = cov + np.dot(diff.T, diff) / float(length)


    print "Trajs: " + str(len(trajs))
    return cov / float(len(trajs))

def sample_iso_cov_lnr(env, lnr, sup, samples, T):
    d = env.action_space.shape[0]
    cov = sample_covariance_lnr(env, lnr, sup, samples, T)
    return np.trace(cov) / float(d) * np.identity(d)

def sample_iso_cov_sup(env, lnr, sup, samples, T):
    d = env.action_space.shape[0]
    cov = sample_covariance_sup(env, lnr, sup, samples, T)
    return np.trace(cov) / float(d) * np.identity(d)

def sample_epsilon_lnr(env, lnr, sup, samples, T):
    surr_loss = statistics.evaluate_agent_disc(env, lnr, sup, T, samples)
    return surr_loss

def sample_epsilon_sup(env, lnr, sup, samples, T):
    loss = statistics.evaluate_sup_disc(env, lnr, sup, T, samples)
    return loss