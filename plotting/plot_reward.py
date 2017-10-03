import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
# color = itertools.cycle(( "#49679E", "#FCB716", "#A0B2D8", "#F68B20", "#2D3956"))
color = itertools.cycle(( "#49679E", "#FCB716", "#A0B2D8", "#F68B20", "#2D3956"))




def main():

    # In the event that you change the sub_directory within results, change this to match it.
    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--iters', required=True, type=int, nargs='+')
    ap.add_argument('--update', required=True, nargs='+', type=int)
    ap.add_argument('--save', action='store_true', default=False)
    
    params = vars(ap.parse_args())
    params['arch'] = [64, 64]
    params['lr'] = .01
    params['epochs'] = 50

    should_save = params['save']
    del params['save']

    plt.style.use('ggplot')

    iters = params['iters']
    ptype = 'surr_loss'
    

    # Behavior Cloning loss on sup distr
    title = 'test_bc'
    ptype = 'sup_reward'
    params_bc = params.copy()
    del params_bc['update']     # Updates are used in behavior cloning
    means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='Supervisor', color='green')

    title = 'test_dart'
    ptype = 'sup_reward'
    params_dart = params.copy()
    means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='Noisy Supervisor', color='green', linestyle='--')

    # BC loss on lnr distr
    title = 'test_bc'
    ptype = 'reward'
    params_bc = params.copy()
    del params_bc['update']     # Updates are used in behavior cloning
    means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='Behavior Cloning', color='red')
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color='red')


    # DART
    title = 'test_dart'
    ptype = 'reward'
    params_dart = params.copy()
    means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='DART', color='blue')
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color='blue')
 


    plt.title(params['envname'])
    plt.legend()
    plt.xticks(iters)
    plt.legend()

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_reward.pdf")
        plt.savefig(save_path + str(params['envname']) + "_reward.svg")
    else:
        plt.show()



if __name__ == '__main__':
    main()


