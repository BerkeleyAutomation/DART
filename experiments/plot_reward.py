import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
color = itertools.cycle(( "#FCB716", "#2D3956", "#A0B2D8", "#988ED5", "#F68B20", 'purple'))



def main():

    # In the event that you change the sub_directory within results, change this to match it.
    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--iters', required=True, type=int, nargs='+')
    ap.add_argument('--update', required=True, nargs='+', type=int)
    ap.add_argument('--save', action='store_true', default=False)
    ap.add_argument('--normalize', action='store_true', default=False)
    
    params = vars(ap.parse_args())
    params['arch'] = [64, 64]
    params['lr'] = .01
    params['epochs'] = 100

    should_save = params['save']
    should_normalize = params['normalize']
    del params['save']
    del params['normalize']

    plt.style.use('ggplot')

    iters = params['iters']
    ptype = 'surr_loss'
    

    # Best supervisor reward
    title = 'test_bc'
    ptype = 'sup_reward'
    params_bc = params.copy()
    del params_bc['update']     # Updates are used in behavior cloning
    means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='Supervisor', color='green')

    sup_means, sup_sems = means, sems
    def normalize(means, sems):
        if should_normalize:
            means = means / sup_means
            sems = sems / sup_means
            return means, sems
        else:
            return means, sems



    # Noisy supervisor reward using DART
    title = 'test_dart'
    ptype = 'sup_reward'
    params_dart = params.copy()
    try:
        means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(iters, means, label='DART Noisy Supervisor', color='green', linestyle='--')
    except IOError:
        pass

    # BC
    title = 'test_bc'
    ptype = 'reward'
    params_bc = params.copy()
    del params_bc['update']     # Updates are used in behavior cloning
    c = next(color)
    try:
        means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(iters, means, label='Behavior Cloning', color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # DAgger
    title = 'test_dagger'
    ptype = 'reward'
    params_dagger = params.copy()
    del params_dagger['update']
    params_dagger['beta'] = .5
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dagger, iters, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(iters, means, label='DAgger', color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # DAgger B
    betas = [.5]
    colors = ['blue', 'red', 'black', 'pink', 'aqua']
    for beta, c in zip(betas, colors):

        title = 'test_dagger_b_beta' + str(beta)
        ptype = 'reward'
        params_dagger_b = params.copy()
        params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
        # c = next(color)
        try:
            means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
            means, sems = normalize(means, sems)
            plt.plot(iters, means, color=c, label=title)
            plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
        except IOError:
            pass

    colors = ['blue', 'red', 'black', 'pink', 'aqua'][::-1]
    for beta, c in zip(betas, colors):

        title = 'test_dagger_b2_beta' + str(beta)
        ptype = 'reward'
        params_dagger_b = params.copy()
        params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
        # c = next(color)
        try:
            means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
            means, sems = normalize(means, sems)
            plt.plot(iters, means, color=c, label=title)
            plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
        except IOError:
            pass
      


    # Isotropic noise
    title = 'test_iso'
    ptype = 'reward'
    params_iso = params.copy()
    params_iso['scale'] = 1.0
    del params_iso['update']
    c = next(color)
    try:
        means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(iters, means, color=c, label='Isotropic')
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # DART
    title = 'test_dart'
    ptype = 'reward'
    params_dart = params.copy()
    c = next(color)
    try: 
        means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(iters, means, label='DART', color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass

    parts = [5, 10, 50, 450]
    for part in parts:
        title = 'test_dart2'
        ptype = 'reward'
        params_dart = params.copy()
        params_dart['partition'] = part
        c = next(color)
        try: 
            means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
            means, sems = normalize(means, sems)
            plt.plot(iters, means, label='DART2_' + str(part), color=c)
            plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
        except IOError:
            pass

    # partitions = [3, 10]
    # colors = ['purple', 'green']
    # for part, c in zip(partitions, colors):
    #     title = 'test_dart3_part' + str(part)
    #     ptype = 'reward'
    #     params_dart = params.copy()
    #     params_dart['partition'] = part
    #     # c = next(color)
    #     try: 
    #         means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
    #         means, sems = normalize(means, sems)
    #         plt.plot(iters, means, label=title, color=c)
    #         plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    #     except IOError:
    #         print "failed to load"
    #         pass




    plt.title("Reward on " + str(params['envname']))
    plt.legend()
    plt.xticks(iters)
    plt.legend()
    if should_normalize:
        plt.ylim(0, 1.05)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_reward.pdf")
    else:
        plt.show()



if __name__ == '__main__':
    main()


