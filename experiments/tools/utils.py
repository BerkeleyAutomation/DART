import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import IPython

def stringify(lst):
    s = ""
    for el in lst:
        s += str(el) + '-'
    return s[:-1]

def clear():
    plt.ioff()
    plt.clf()
    plt.cla()
    plt.close()

def score(est, X, y):
    preds = est.predict(X)
    loss = np.sum(np.square(y - preds))
    res = np.sum(np.square(y - np.mean(y, axis=0)))
    return 1 - loss/res

def stringify(param):
    if isinstance(param, list):
        s = ""
        for el in param:
            s += str(el) + '-'
        return s[:-1]
    else:
        return str(param)



def trajs2data(trajs):
    for i in range(len(trajs)):
        traj = trajs[i]
        states, controls = zip(*traj)
        if i == 0:
            X = np.array(states)
            y = np.array(controls)
        else:
            X = np.vstack((X, np.array(states)))
            y = np.vstack((y, np.array(controls)))
    return X, y




def generate_dir(title, sub_dir, input_params):
    d = 'results/' + sub_dir + '/' + input_params['envname'] + '.pkl/' + title + '/'
    keys = sorted(input_params.keys())
    for key in keys:
        param = stringify(input_params[key])
        d += key + param + '_'
    return d[:-1]


def generate_plot_dir(title, sub_dir, input_params):
    d = generate_dir(title, sub_dir, input_params)
    return d + '_plot/'

def generate_data_dir(title, sub_dir, input_params):
    d = generate_dir(title, sub_dir, input_params)
    return d + '_data/'


def extract_data(params, iters, title, sub_dir, ptype):
    means, sems = [], []
    for it in iters:
        params['it'] = it

        path = generate_data_dir(title, sub_dir, params) + 'data.csv'
        data = pd.read_csv(path)
        arr = np.array(data[ptype])

        mean, sem = np.mean(arr), scipy.stats.sem(arr)
        means.append(mean)
        sems.append(sem)
    return np.array(means), np.array(sems)


def extract_time_data(params, iters, title, sub_dir):
    params['iters'] = iters
    path = generate_data_dir(title, sub_dir, params) + 'time_tests.txt'
    data = pd.read_csv(path)
    arr = np.array(data['time'])
    mean, sem = np.mean(arr), scipy.stats.sem(arr)
    return mean, sem



def filter_data(params, states, i_actions):
    """
        Filter the data, choosing only 50 samples
        as described in Ho and Ermon, 2016
    """
    T = params['t']
    k = np.random.randint(0, T/50)
    left_out_states, left_out_actions = [], []
    for j in range(T/50):
        if not j == k:
            left_out_states += states[j::T/50]
            left_out_actions += i_actions[j::T/50]

    new_states, new_i_actions = states[k::T/50], i_actions[k::T/50]

    return new_states, new_i_actions, (left_out_states, left_out_actions)


