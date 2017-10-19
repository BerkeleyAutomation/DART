import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
color = itertools.cycle(( "#FCB716", "#49679E", "#F68B20",))



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
    ptype = 'sup_loss'
    params_bc = params.copy()
    del params_bc['update']     # Updates are used in behavior cloning
    c = next(color)
    means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
    plt.plot(iters, means, color=c, linestyle='--')

    ptype = 'surr_loss'
    means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='Behavior Cloning', color=c)
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)


    # Rand
    title = 'test_rand'
    ptype = 'sup_loss'
    params_rand = params.copy()
    params_rand['prior'] = 1.0      # You may adjust the prior to whatever you chose.
    del params_rand['update']
    c = next(color)
    means, sems = utils.extract_data(params_rand, iters, title, sub_dir, ptype)
    plt.plot(iters, means, color=c, linestyle='--')

    ptype = 'surr_loss'
    means, sems = utils.extract_data(params_rand, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='Rand Loss', color=c)
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)

    ptype = 'sim_err'
    means, sems = utils.extract_data(params_rand, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='Rand Sim. Err.', color=c, linestyle=':')


    # DART
    title = 'test_dart'
    ptype = 'sup_loss'
    params_dart = params.copy()
    c = next(color)
    means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
    plt.plot(iters, means, color=c, linestyle='--')
    
    ptype = 'surr_loss'
    means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='DART', color=c)
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)

    ptype = 'sim_err'
    means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
    plt.plot(iters, means, label='DART Sim. Err.', color=c, linestyle=':')



    plt.title("Loss on " + str(params['envname']))
    plt.legend()
    plt.xticks(iters)
    plt.legend(loc='upper right')

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_loss_rand.pdf")
    else:
        plt.show()



if __name__ == '__main__':
    main()


