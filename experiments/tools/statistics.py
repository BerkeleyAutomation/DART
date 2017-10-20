import numpy as np
import scipy.stats
import random

def eval_agent_statistics_cont(env, agent, sup, T, num_samples=1):
    """
        evaluate loss in the given environment along the agent's distribution
        for T timesteps on num_samples
    """
    losses = []
    for i in range(num_samples):
        # collect trajectory with states visited and actions taken by agent
        tmp_states, _, tmp_actions, _ = collect_traj(env, agent, T)
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (sup_actions - tmp_actions) ** 2.0
        # compute the mean error on that trajectory (may not be T samples since game ends early on failures)
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))

    # compute the mean and sem on averaged losses.
    return stats(losses)

def eval_sup_statistics_cont(env, agent, sup, T, num_samples=1):
    """
        Evaluate loss on the supervisor's trajectory in the given env
        for T timesteps
    """
    losses = []
    for i in range(num_samples):
        # collect states made by the supervisor (actions are sampled so not collected)
        tmp_states, _, _, _ = collect_traj(env, sup, T)

        # get inteded actions from the agent and supervisor
        tmp_actions = np.array([ agent.intended_action(s) for s in tmp_states ])
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (sup_actions - tmp_actions) ** 2.0

        # compute the mean error on that traj
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))

    # generate statistics, same as above
    return stats(losses)


def eval_sim_err_statistics_cont(env, sup, T, num_samples = 1):
    losses = []
    for i in range(num_samples):
        tmp_states, int_actions, taken_actions, _ = collect_traj(env, sup, T)
        int_actions = np.array(int_actions)
        taken_actions = np.array(taken_actions)
        errors = (int_actions - taken_actions) ** 2.0
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))
    return stats(losses)


def eval_rewards(env, agent, T, num_samples=1):
    reward_samples = np.zeros(num_samples)
    for j in range(num_samples):
        _, _, _, reward = collect_traj(env, agent, T)
        reward_samples[j] = reward
    return np.mean(reward_samples)


def stats(losses):
    if len(losses) == 1: sem = 0.0
    else: sem = scipy.stats.sem(losses)

    d = {
        'mean': np.mean(losses),
        'sem': sem
    }
    return d

def ste(trial_rewards):
    if trial_rewards.shape[0] == 1:
        return np.zeros(trial_rewards.shape[1])
    return scipy.stats.sem(trial_rewards, axis=0)

def mean(trial_rewards):
    return np.mean(trial_rewards, axis=0)


def mean_sem(trial_data):
    s = ste(trial_data)
    m = mean(trial_data)
    return m, s


def evaluate_lnr_cont(env, agent, sup, T, num_samples = 1):
    stats = eval_agent_statistics_cont(env, agent, sup, T, num_samples)
    return stats['mean']

def evaluate_sup_cont(env, agent, sup, T, num_samples = 1):
    stats = eval_sup_statistics_cont(env, agent, sup, T, num_samples)
    return stats['mean']

def evaluate_sim_err_cont(env, sup, T, num_samples = 1):
    stats = eval_sim_err_statistics_cont(env, sup, T, num_samples)
    return stats['mean']


def collect_traj(env, agent, T, visualize=False):
    """
        agent must have methods: sample_action and intended_action
        Run trajectory on sampled actions
        record states, sampled actions, intended actions and reward
    """
    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()

    reward = 0.0

    for t in range(T):

        a_intended = agent.intended_action(s)
        a = agent.sample_action(s)
        next_s, r, done, _ = env.step(a)
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if done == True:
            break
            
    return states, intended_actions, taken_actions, reward

def collect_traj_beta(env, sup, lnr, T, beta, visualize=False):

    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()

    reward = 0.0
    count = 0
    for t in range(T):

        a_intended = lnr.intended_action(s)

        if random.random() > beta:
            a = lnr.sample_action(s)
        else:
            a = sup.sample_action(s)
            count += 1
        
        next_s, r, done, _ = env.step(a)
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if done == True:
            break
     
    print "Beta: " + str(beta), "empirical beta: " + str(float(count) / (t + 1))       
    return states, intended_actions, taken_actions, reward

