import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
color = itertools.cycle(( "#FCB716", "#2D3956", "#A0B2D8", "#988ED5", "#F68B20"))



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
    params['epochs'] = 100

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
    try:
        means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
        plt.plot(iters, means, color=c, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
        plt.plot(iters, means, label='Behavior Cloning', color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass



    # DAgger
    beta = .5
    title = 'test_dagger'
    ptype = 'sup_loss'
    params_dagger = params.copy()
    params_dagger['beta'] = .5      # You may adjust the prior to whatever you chose.
    del params_dagger['update']
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dagger, iters, title, sub_dir, ptype)
        plt.plot(iters, means, color=c, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_dagger, iters, title, sub_dir, ptype)
        plt.plot(iters, means, label='DAgger', color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # DAgger B
    beta = .5
    title = 'test_dagger_b'
    ptype = 'sup_loss'
    params_dagger_b = params.copy()
    params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
        plt.plot(iters, means, color=c, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
        plt.plot(iters, means, label='DAgger-B', color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # Isotropic noise
    title = 'test_iso'
    ptype = 'sup_loss'
    params_iso = params.copy()
    params_iso['scale'] = 1.0
    del params_iso['update']
    c = next(color)
    try:
        means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
        plt.plot(iters, means, color=c, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
        plt.plot(iters, means, label='Isotropic Noise', color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass



    # DART
    partition = 450
    title = 'test_dart'
    ptype = 'sup_loss'
    params_dart = params.copy()
    params_dart['partition'] = partition
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
        plt.plot(iters, means, color=c, linestyle='--')
        
        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
        plt.plot(iters, means, label='DART ' + str(partition), color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass



    plt.title("Loss on " + str(params['envname']))
    plt.legend()
    plt.xticks(iters)
    plt.legend(loc='upper right')

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_loss.pdf")
    else:
        plt.show()



if __name__ == '__main__':
    main()


